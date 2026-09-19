# Implementation plan: `ttu-tower-processing` 2.0.0

For a Claude Sonnet implementation session, written by Claude Opus 5 (architect) with Elliott
Walker. The session has no access to the design conversation; these documents are
self-contained.

| Document | Role |
|---|---|
| `overview.md` | A one-page summary of the design, with architecture and data-flow diagrams. It restates the other documents, which take precedence. |
| `plan.md` (this file) | How to build it: working norms, architecture, conventions, and phases 0–10 with contracts, algorithms and acceptance tests. |
| `design-reference.md` | What the pipeline does and why each choice was made. Read it fully before Phase 0. Everything in it is settled. |
| `config-reference.md` | Every parameter: default, provenance, validation rules; the config template. |
| `output-schema.md` | Every output table and column: the contract between stages and for downstream users. |
| `validation.md` | Test principles, synthetic data generators, and the first-run validation procedures. |
| `definitions.md` | The terms these documents use. |

Reading order: overview.md, then this §1, then definitions.md, then design-reference.md, then
the rest of this file, then the others as each phase needs them.

---

## 1. Working norms for the implementation session

### 1.1 Process
- **Work phase by phase, in order.** At the end of each phase, stop and report to Elliott (§1.5),
  then wait for his go-ahead before starting the next. He reviews all code.
- **Don't commit or push unless Elliott asks.** He keeps focused, single-purpose commits and
  will say when and how. `docs/` is tracked, so an edit to one of these documents is committed
  like any other change, when he asks for it.
- **Other repositories are read-only references:** `C:\Users\ellwalke\Code\old-tower-processing`
  (the old pipeline) and `C:\Users\ellwalke\Code\windprofiles`. Never modify, install or run
  their pipelines. The single exception is generating the regression fixture in Phase 2, which
  reads one raw file with the old code; Elliott has approved that.
- **Ask before running anything that writes pipeline results** (including `--test` runs on real
  data), runs for more than a few minutes, or touches another repo, a remote, or GitHub. Unit
  tests and short read-only checks don't need permission; tests never touch the real ttu-tower
  home (§3.10). Elliott has already approved the real-data runs this plan names (Phase 6's
  `--test` run, Phase 10's pilot and production runs): say when you start one, and don't widen
  it.
- **Python 3.13 in `.venv`** (`py -3.13 -m venv .venv` if it's missing). Install with
  `.venv\Scripts\python.exe -m pip install -e ".[dev]"`.

### 1.2 Code style
- **Readability first; no bloat, no clutter.** Small, pure functions over numpy arrays; the
  orchestration layers are thin.
- **Comments are terse and sparse**: state only what would otherwise be unclear (a formula's
  source, a non-obvious invariant). **Never put conversation-specific or process notes in code
  comments** (no "per the design discussion", no roadmap notes, no "Phase 3 will…"). One-line
  docstrings on public functions; longer only where the contract is subtle.
- Type hints on public functions. f-strings. `pathlib`. No wildcard imports.
- Numerics: vectorize with numpy; loop over modes, rungs or slots only where the count is
  small. float64 for all computation (raw files are float32).
- **Every uncertain numeric parameter is config**, never a hardcoded constant (config-reference.md).
  Site facts (heights, tilt tables, header maps) live in `constants.py`, physical constants in
  `physics/constants.py`.
- **`math/` and `physics/` are libraries:** pure functions of numpy arrays and floats that know
  nothing of the pipeline (no config, grids, masks or tables). `math/` uses only numpy;
  `physics/` may use `math/`, never the reverse. Stage modules keep only the pipeline-specific
  parts (block assignment, pooling, coverage) and call these. Import their modules explicitly
  (`from ttu_tower.math import polar`), never `from ttu_tower import math`, which would shadow
  the standard library's `math`.
- Don't edit `README.md` unless Elliott asks.

### 1.3 The design documents
- design-reference.md is settled. If implementation exposes a concrete problem with a decision,
  stop and raise it with Elliott; don't work around it silently.
- If a contract in `output-schema.md` or a key in `config-reference.md` must change, change the
  document in the same phase and list the change in the phase report.
- **Old-pipeline behavior is history, not validation** (design-reference M): port its code where
  this plan says so, but "the old code did X" is not evidence that X is right.

### 1.4 Testing
- `pytest` from the repo root. Real-data tests are marked `integration` and skipped unless
  `--integration` is given. validation.md §1 gives the test principles and §2 the synthetic
  data generators.
- Each phase's acceptance tests must pass before the phase report. A failing test is reported
  with its output, never weakened to pass.

### 1.5 Phase report (end of every phase)
1. What was built (modules, public functions).
2. Test results (counts; all passing, or the failures with output).
3. Deviations from this plan and why; document edits made.
4. Timings where relevant (from Phase 6 on).
5. Questions or problems for Elliott.

---

## 2. Architecture

### 2.1 Data flow

```
raw parquet files (30 min, 50 Hz, source units)
   │  ttu-files: file table (bad_records → offset rule → collision guard)
   ▼
PRIMARY  (per (batch, boom) work unit, streaming over files)
   Stage A  per file:  load boom columns → units → bounds → triplet coupling → tilt → rotate to (ue, vn, w)
   Stage B  per file, on a span with ±10 min margins (one file behind Stage A):
            despike → fill ≤1 s gaps → higher moments → resolution/dropouts → slow smoothing
            → vpts → first-layer masks → slot means → direction/bounce → final masks
            → floor-grid sums & finest-block sums (per variant) → flags, coverage, means, gusts
   Products per slot (two files behind Stage B):
            MRD detection outputs (80-min window) · statistics ladder (8 rungs) · ITS & TE per rung
   ▼  tables: files, slots, slot_boom, coverage, means, slot_qc, flags, ladder, ladder_coverage, mrd, mrd_frame
SECONDARY  (per batch)
   τ detection (heat, momentum) → τ selection → pressure factors → selected-τ statistics
   → derived quantities per variant
   slow-sensor thermodynamics · sun elevation / night
   ▼  tables: tau, tau_selected, boom_stats, boom_labels, slow, slot_stats
TERTIARY  (per batch)
   mesonet merge → boom-level filtering (NaN) → Ri_b, lapse, veer → profile fits
   ▼  tables: boom_final, tau_final, pairs, profile, slot_final, filter_log; wide/<month>.parquet
POST  stability classification helpers; yield-by-stability report
```

### 2.2 Package layout

```
pyproject.toml
configs/
  templates/oneyear_TEMPLATE.toml
  oneyear.toml                copy of the template for running (git-ignored)
src/ttu_tower/
  __init__.py                 __version__; re-exports load_results, find_run, list_runs
  results.py                  loading results by run tag (§3.10)
  constants.py                site, booms, heights, tilt tables, header maps, source units,
                              file and window constants, P_REF
  timegrid.py                 epoch, sample/slot/half-hour indices, block grids, fractional block sums
  flags.py                    test registry, interval builder, FlagStore queries
  schema.py                   output table definitions and dtype casting
  logs.py                     queue-based multiprocess logging
  config/
    __init__.py               load_config()
    model.py                  frozen dataclasses (one per config section)
    load.py                   TOML → dataclasses, defaults, validation
    hashing.py                per-stage config hashes
  io/
    rawfiles.py               filename parsing, timing rule, file table
    load.py                   read one boom's columns from a converted file
    convert.py                raw csv/gz/zip → parquet (ported)
    store.py                  fragments, atomic writes, manifests, run_meta, read_table
    runs.py                   ttu-tower home, run registry, run-directory resolution
  math/
    polar.py                  bearing ↔ vector, angular distances, Yamartino σ_θ
    stats.py                  autocovariance, e-folding integral, linear detrend, moments
    fits.py                   linear, weighted, power-law and log fits
  physics/
    constants.py              R, c_p, p0, ε, g0, κ, Ω
    thermo.py                 pressure (ISA), humidity (Magnus), temperatures, VPT
    most.py                   u*, L, z/L, Businger–Dyer, neutral log-law fits
    richardson.py             bulk Richardson number
    turbulence.py             TI, TKE, ILS, transport efficiency, anisotropy
    earth.py                  local gravity, Coriolis parameter
  primary/
    stage_a.py
    despike.py
    gaps.py
    qc_windows.py             resolution/dropouts, higher moments
    slow.py                   slow-sensor smoothing, vpts
    masks.py                  families, first/second layer, final masks
    blocks.py                 floor-grid and finest-block family sums
    stage_b.py                composes Stage B for one file span
    mrd.py                    MRD modes, detection outputs
    ladder.py                 rung statistics from finest-block sums
    rawrung.py                ITS and TE per rung from samples
    slotstats.py              slot means, maxima, gusts
    products.py               per-slot products for one file
    partition.py              processing batches from the file table
    stream.py                 per-(batch, boom) streaming driver
    runner.py                 work units, pool, manifest, resume
    report.py                 primary QC/yield report
  secondary/
    detect.py  select.py  derived.py  slow.py  sun.py  runner.py
  tertiary/
    mesonet.py  filtering.py  multiboom.py  profiles.py  wide.py  runner.py
  post/
    classify.py  report.py
  validation/
    synthetic.py  checks.py
  cli/
    files.py  primary.py  secondary.py  tertiary.py  runall.py  report.py  validate.py  runs.py  convert_parquet.py
tests/
  conftest.py
  unit/<mirrors src/ttu_tower>/test_*.py
  integration/test_*.py
  fixtures/
```

### 2.3 Commands

| Command | Does |
|---|---|
| `ttu-files CONFIG [--allow-non-parquet]` | Build and print the file table (status counts); write `primary/files.parquet`. |
| `ttu-primary CONFIG [--nproc N] [--test] [--redo-failures] [--force] [--allow-non-parquet]` | Run primary. |
| `ttu-secondary CONFIG [--test] [--force]` | Run secondary. |
| `ttu-tertiary CONFIG [--test] [--force]` | Run tertiary. |
| `ttu-runall CONFIG [--skip-primary] [--nproc N] [--test]` | Run the stages in sequence (subprocesses, like the old `runall`). `--nproc` goes to primary. |
| `ttu-report CONFIG [--stability]` | Write the QC/yield reports (`<run dir>/reports/`). |
| `ttu-validate CONFIG --check NAME` | Run a first-run validation check (validation.md §3). |
| `ttu-runs [--test]` | List the registered runs: tag, run directory, whether it exists, stages present (§3.10). |
| `ttu-convert-parquet …` | Port of the old converter CLI, unchanged in behavior. |

- Only primary runs in parallel (`--nproc`, a `spawn` pool of work units). Secondary and
  tertiary run their batches serially in one process: they take minutes per year (§6).
- `--test` processes only the first batch and writes to a separate test run directory
  (§3.10).
- `--force` overrides the config-hash guard by clearing the stage's outputs and every later
  stage's first. It also re-registers a tag whose link points to another existing run
  directory (§3.10).
- `--allow-non-parquet` lets raw `.csv`/`.csv.gz`/`.zip` files be read directly (Phase 5).
  Without it, finding such a file stops the run with the message "N unconverted raw files in
  <dir>; run `ttu-convert-parquet` first, or pass --allow-non-parquet (slow; for small runs)".

---

## 3. Conventions (normative for all phases)

### 3.1 Time, indices, grids
- `EPOCH = pd.Timestamp("2000-01-01 00:00", tz="Etc/GMT+6")` (in `timegrid.py`);
  `SAMPLE_HZ = 50` (in `constants.py`, with the other file and window constants).
- Sample index `g` = (t − EPOCH)/20 ms, int64. Slot k = samples `[30000k, 30000(k+1))`;
  half-hour (file) h = samples `[90000h, 90000(h+1))` = slots 3h, 3h+1, 3h+2.
- **Detection window W = 80 min and floor level K** (`mrd.floor_level`, default 15). Numbers in
  these documents are for K = 15; code derives them from K.
- **Block grids** are anchored at g = 0 and have rational block lengths `num/den` samples:

  | Grid | Length (samples) | num/den | Per slot |
  |---|---|---|---|
  | floor (K = 15, W = 80 min) | 7.32421875 | 1875/256 | 4096 |
  | finest (9.375 s) | 468.75 | 1875/4 | 64 |
  | despike stride centers | 1500 | 1500/1 | 20 |
  | resolution/dropout windows | start every 2500, length 5000 | — | — |
  | gust blocks (p s) | 50p | 50p/1 | 600/p |

  Slow-sensor smoothing has no grid: it is a kernel centered on every sample (Phase 2).

- **Fractional block sums** (`timegrid.block_sums`). Sample g covers `[g, g+1)`; block b covers
  `[b·num/den, (b+1)·num/den)`. In integer units of 1/den: the sample is `[g·den, (g+1)·den)`
  and the block is `[b·num, (b+1)·num)`. With k = (g·den) // num, the sample's share in block k
  is `min((k+1)·num, (g+1)·den) − g·den` (over den), and the rest goes to block k+1. Since
  num > den, a sample spans at most two blocks. Accumulate with `np.bincount(…, weights=…,
  minlength=…)` once for each share. For any column x with per-sample weight w (0/1 usable),
  this gives Σ share·w·x per block and the weight sum Σ share·w. Everything is exact integer
  bookkeeping, and a block's result depends only on its own samples (translation invariance).
- ITS/TE use whole samples: sample g belongs to block `((2g+1)·den) // (2·num)` (majority rule).
- Rung τ_j = 9.375·2^j s, j = 0..7 ↔ 2^j finest blocks (j = 7 is the 20-min rung, 128 finest
  blocks centered on the slot: `[64k − 32, 64k + 96)`).
- **MRD modes are labeled by averaging time scale** (VM06): mode i = 1..15 has scale
  P_i = 0.29296875·2^(i−1) s = 2^i floor blocks. Its value is the mean, over the window's
  2^(15−i) blocks of width P_i, of ΔxΔy/4, where Δ is the difference between the block's two
  halves (each 2^(i−1) floor blocks).
  - Mode i ≥ 6 has P_i = τ_(i−6), a ladder rung; P_14 = 40 min and P_15 = 80 min are above the
    ladder.
  - A reversal at mode r gives τ = P_(r−1).
  - The detection window of slot k is floor blocks `[4096k − 14336, 4096k + 18432)` (32,768
    blocks, centered on the slot).
- **Output time zone:** internal indices never depend on it. Timestamps written to outputs are
  tz-aware in `[output].timezone` (`Etc/GMT+6` or `UTC`), and `[period]` strings are read in it.
  Slot boundaries are identical in both zones (whole-hour offset).

### 3.2 Units and the per-sample chain (Stage A)
Source units: speeds mph, ts and t °F, rh %, p inHg. Conversions, ported exactly from
`windprofiles/processing/units.py`: m/s = mph / 2.23694; K = (°F − 32)·5/9 + 273.15;
fraction = % / 100; kPa = inHg × 3.38639.

Order: **units → bounds → triplet coupling → tilt → rotation** (design-reference C).
Implement Stage A as an explicit ordered list of per-sample step functions. A future
angle-of-attack correction is inserted between coupling and tilt (design-reference L), with no
other change.
- Raw sonic components are `u` = north and `v` = west components of the wind's FROM-vector,
  and `w` = up.
- Tilt runs in that raw (N, W, up) frame (the old `fix_sonic_tilt_misalignment`).
- Rotation: `ue = v_raw`, `vn = −u_raw`, `w = w` (the old `compute_winds`). This yields true
  (east, north) blows-toward components.

### 3.3 Frames and directions
- Wind direction = FROM-bearing via `math.polar.vector_to_bearing(ue, vn)`, the single source
  of truth; `bearing_to_vector` is its exact inverse. No `flip`-style parameter anywhere.
- **Streamwise rotation** of a block with mean vector (ū_e, v̄_n): φ = atan2(v̄_n, ū_e);
  u = ue·cosφ + vn·sinφ; v = −ue·sinφ + vn·cosφ (v points left of the mean wind); w unchanged.
  For a 3×3 Earth-frame covariance C: C' = Q C Qᵀ with Q = [[cosφ, sinφ, 0], [−sinφ, cosφ, 0],
  [0, 0, 1]].
- No pitch rotation: sensor tilt is handled by the fixed Kelly & Ennis correction, and mean w
  is reported as a diagnostic.

### 3.4 Variables, masks, families
- Buffer variables per boom: `ue, vn, w, ts, t, rh, p`, derived `vpts` (and per-sample `ws` when
  needed).
- **Triplet coupling:** any sample removed or NaN in one of ue, vn, w is NaN in all three.
- **First-layer usable mask** per variable: finite after ≤ 1-s gap filling, and outside the
  intervals of every test in `unusable_tests` except `direction` and `bounce`.
- **Final usable mask** = first layer, and not inside a `direction`/`bounce` slot, if those
  tests are in `unusable_tests`; they apply to ue, vn, w, ts.
- **vpts** = ts·(100/P_REF[b])^R_CP with a fixed per-boom reference pressure (design-reference
  I), so the **vpts mask = ts mask**. Secondary converts the heat quantities to the measured slot
  pressure (Phase 7).
- **Families** (combined masks): `momentum` = ue ∧ vn ∧ w; `heat` = w ∧ vpts; `ts` = ts.
- **`mrd_unexcised` series and weights:** the within-run-filled series (gaps ≤ 10 min), with
  weight 1 wherever it is finite, masks ignored.

### 3.5 Coverage threshold
One parameter, c = `qc.min_coverage`. A partial window is valid if
usable weight / nominal length ≥ c. It applies to floor blocks and MRD half-blocks at every
scale, τ-blocks and the 20-min window, despiking reference windows, smoothing kernels, gust
blocks, higher-moment slots, and first-layer slot means for the second layer.

### 3.6 Flags
The test registry (`flags.TESTS`) records each test's name, kind, grain and variables
(output-schema.md §3.7). Flags are produced as intervals of global sample indices. Primary
builds masks from its own intervals; consumers query `FlagStore`. Query-time thresholds
(`max_bounds_fraction`, `max_spike_fraction`) exist only in tertiary.

### 3.7 Invariance requirements (make them true by construction)
- **Seam invariance:** Stage B results for a file's core are identical whether the span is the
  file ± 10 min or a whole concatenated run. Products for a slot are identical whatever batches
  the timeline is split into.
- **Parallel = serial:** outputs don't depend on `nproc` or scheduling order.
- How: every windowed computation is anchored to the global grids of §3.1 and reads only
  samples within its reach. Nothing accumulates across a span (no running sums whose rounding
  depends on where a span starts). The Stage B margin (10 min) exceeds every reach
  (config-reference.md cross-key rules).
- Tests compare floats with `rtol=1e-12` and discrete outputs (masks, flags, statuses, counts)
  exactly. A discrete mismatch must be traced, not tolerated.

### 3.8 Config, hashes, run directories
- `load_config(path) -> Config` reads TOML, fills defaults, validates (config-reference.md), and
  rejects unknown keys.
- A stage's directory is `<run dir>/<stage>/`, with the run directory resolved and registered
  as in §3.10.
- Stage hashes: SHA-256 (first 16 hex chars) of canonical JSON (sorted keys, no whitespace) of
  the sections each stage depends on, plus the package's major.minor version:
  - `primary`: `files`, `qc`, `mrd`, `ladder`, `primary` (without `nproc`), `output`,
    `paths.raw_dirs`, `period`.
  - `secondary`: `secondary` + the primary hash.
  - `tertiary`: `tertiary`, `paths.mesonet_dir` + the secondary hash.
- Each stage writes `config.toml` (a copy), `config.resolved.json` and `run_meta.json`
  (output-schema.md §6) into its directory before any data. On start it compares the hash with
  any existing `run_meta.json`, and refuses to run on a mismatch unless `--force`.

### 3.9 Logging and errors
- Port `windprofiles/user/logs.py`: JSON-lines formatter (Cutelog-compatible), a queue listener
  in the main process, a `QueueHandler` in workers, and the warnings bridge. Logs go to
  `<run dir>/<stage>/logs/<stage>.log`; the console gets progress (tqdm) and a summary.
- Primary's pool uses the `spawn` multiprocessing context on every platform (Windows now, Linux
  on HPCC later). Worker functions are top-level and picklable. Secondary and tertiary use no
  pool.
- A work unit that raises is caught, logged with its traceback, and marked `failed` in the
  manifest, and the run continues. Expected data conditions (missing files, empty slots, NaN
  runs) are statuses, never exceptions.
- **Levels:**
  - `INFO`: run, batch and unit start/finish and summaries;
  - `WARNING`: data anomalies (a collision, a bad-length file, a unit with no usable data);
  - `ERROR`: failures, with the traceback;
  - `DEBUG`: per-file and per-slot detail, off by default.
- **Context fields** on every JSON record: `stage`, `unit`, `batch`, `boom`, `half_hour` (null
  where not applicable). Set them with a `LoggerAdapter` or a filter, not by hand in each call.
- **One summary record per work unit** (per batch in secondary and tertiary), logged at `INFO`
  and also stored in the unit's manifest entry:
  - files loaded and missing;
  - slots by status;
  - flagged-sample fractions by test;
  - how many slots have `unexcised_computed`;
  - wall time per step (Stage A, Stage B, detection outputs, ladder, rawrung, writing).

  The Phase 6 timing report is built from these.
- **numpy warnings** (`RuntimeWarning` from empty means, divisions and so on) are caught per unit
  with `np.errstate` and `warnings.catch_warnings`, counted by message, and reported as counts
  in the unit summary, not logged one by one.
- **`run_summary.json`** per stage, next to `run_meta.json`, written at the end of a run: unit
  or batch counts by status, totals of the unit summaries, total and per-step wall time, the
  package version and the config hash.

### 3.10 Where results live: the ttu-tower home and the run registry
Results never go inside the package or the repository. Every run is registered in a small
directory in the user's home, found the same way on every machine, and results are loaded by
run tag through it. `io/runs.py` implements this; `results.py` is the public loading API.

**The ttu-tower home**
```
~/.ttu-tower/
  ttu-tower-home.json          signature
  runs/<tag>.json              one link file per run (runs/testing/<tag>.json for --test runs)
  results/<tag>/               run directories, when [paths].results_dir is empty
  results/testing/<tag>/
```
- **Signature:** `ttu-tower-home.json` holds `{"signature": "ttu-tower-processing home",
  "created": "<ISO time>"}`. A directory is a ttu-tower home if and only if this file exists and
  its `signature` value matches.
- **`find_home(user_home=None) -> Path`** (`user_home` defaults to `Path.home()`; the argument
  exists for tests):
  1. If the environment variable `TTU_TOWER_HOME` is set, use exactly that path: adopt it if it
     has the signature, initialize it if it doesn't exist or is an empty directory, and
     otherwise stop with an error. A foreign directory is never adopted.
  2. Otherwise the candidates are `.ttu-tower`, `.ttu-tower0`, `.ttu-tower1`, … in the user's
     home. List the existing entries matching `^\.ttu-tower(\d*)$` (one directory listing), and
     take the first in that order that has the signature. Considering every existing
     candidate, rather than stopping at the first missing name, means that deleting
     `.ttu-tower0` doesn't hide `.ttu-tower1`.
  3. If none has it, create the first candidate name that doesn't exist and write the signature
     (atomically). If that isn't `.ttu-tower`, log a WARNING saying why ("~/.ttu-tower exists
     but isn't a ttu-tower home; created ~/.ttu-tower0").

**Run directory and registration**
- The **tag** (config `tag`, default the config file's stem) identifies a run. A config's
  **run directory** is `<R>/<tag>/`, or `<R>/testing/<tag>/` with `--test`, where R is
  `[paths].results_dir` if it is set, else `<home>/results`.
- **`register_run(cfg, config_path, test, force) -> Path`** runs in the main process of every
  command that writes into the run directory, before it writes anything, and returns the run
  directory. The link file
  `<home>/runs/<tag>.json` (`runs/testing/<tag>.json` for test runs) holds absolute paths:
  ```json
  {"tag": "oneyear", "run_dir": "D:/tower-results/oneyear",
   "config": "C:/Users/ellwalke/Code/ttu-tower-processing/configs/oneyear.toml",
   "registered": "2026-10-02T14:03:11-06:00", "package_version": "2.0.0"}
  ```
  One path is enough, since the run directory holds every stage. The check:
  - no link: write it;
  - a link to the same run directory: nothing to do;
  - a link to a directory that no longer exists: re-point it, with a WARNING;
  - a link to another existing directory: stop with "tag 'oneyear' is registered to <path>;
    use another tag, or pass --force to re-register it (the old results are not deleted)".

  Every writing command runs it, so the registry and the config can't disagree. Registering at
  the start rather than the end keeps an interrupted run findable; each stage's `run_meta.json`
  and `run_summary.json` say how far it got.
- The pipeline commands take a CONFIG and resolve the run directory as above. Loading by tag is
  for everything downstream: notebooks, `ttu-windprofiles`, `ttu-runs`. Workers never touch the
  home or the registry.
- Deleting a run is manual: remove its run directory and its link file. Moved results are
  re-pointed by editing `run_dir` in the link file, which is plain JSON.

**Loading** (`ttu_tower.results`, re-exported from `ttu_tower`). This replaces the old
`io.final.load_results`, which loaded the wide tertiary file located from a config.
- `find_run(tag, *, test=False) -> Path`: the run directory from the link file. Errors say what
  is wrong: an unknown tag (the message lists the registered tags), or a run directory that is
  missing (the message gives the registered path and says the link's `run_dir` can be edited if
  the results were moved).
- `load_results(tag, table="wide", *, test=False, columns=None, filters=None) -> DataFrame`:
  any table of output-schema.md, by name. Table names are unique across stages and
  `schema.TableSpec` records each table's stage and layout (fragment directory or single file),
  so the path follows from the name. Fragments are read with `store.read_table`; `"wide"`
  concatenates the monthly files in order. It warns if that stage has no `run_summary.json`,
  which means the run didn't finish.
- `list_runs(*, test=False) -> DataFrame`: tag, run directory, whether it exists, the stages
  present (those with a `run_meta.json`), registration time. `ttu-runs` prints it.

---

## 4. Phases

Each phase lists its goal, modules, specification and acceptance tests, then ends with the
§1.5 report.

### Phase 0: Scaffolding

**Goal:** an installable, testable, configurable package skeleton.

**Deliverables**
- `pyproject.toml`:
  - setuptools with the `src` layout;
  - `name = "ttu-tower-processing"`, `version = "2.0.0"`, `requires-python = ">=3.13,<3.14"`;
  - dependencies `numpy`, `pandas`, `pyarrow`, `astral`, `tqdm`;
  - extras `mesonet = ["wtxmeso @ git+https://github.com/Intergalactyc/wtxmesonet@main"]` and
    `dev = ["pytest", "scipy"]`;
  - the `[project.scripts]` of §2.3;
  - `[tool.pytest.ini_options]` with `testpaths = ["tests"]` and an `integration` marker.
- `.gitignore`: add `configs/*.toml` (the configs for running; `configs/templates/` is
  tracked). The existing `*.ini` lines can stay.
- `configs/templates/oneyear_TEMPLATE.toml`, copied from config-reference.md. When it loads and
  passes the acceptance tests below, copy it to `configs/oneyear.toml`, the config for running.
  If a later phase changes the template, make the same change in `oneyear.toml`, keeping any
  settings Elliott has changed there.
- `constants.py`, ported from the old `definitions.py`:
  - booms, heights (`HEIGHTS: dict[int, float]`);
  - site latitude, longitude and elevation; `SOURCE_TIMEZONE = "Etc/GMT+6"`;
  - `TILT_ANGLES`, `TILT_AXES`;
  - `SOURCE_HEADERS_OLD/NEW`, `HEADER_MAP_OLD/NEW`, `SOURCE_UNITS` (for the converter);
  - `SAMPLE_HZ`, `ROWS_PER_FILE = 90_000`, `DETECTION_WINDOW_MIN = 80`,
    `STAGE_B_MARGIN_S = 600`;
  - `P_REF: dict[int, float]`, the per-boom reference pressures: `physics.thermo.isa_pressure`
    at (1014 m + boom height), computed once from the heights.

  Drop the plotting and legacy tables (`FIGVARS`, `AUTOCORR_*`, `STABILITY_COLUMNS`, …).
- **`math/` and `physics/`**: the ported functions of §5, as numpy functions (the originals
  take lists or pandas Series), with their windprofiles tests adapted. Changes from the
  originals:
  - e_s and the dewpoint use the Magnus form with the AERK coefficients (design-reference I):
    e_s(T) = 0.61094·exp(17.625·T/(T + 243.04)) kPa with T in °C, and T_d is its exact
    inverse;
  - new `thermo.isa_pressure(z)` = 101.325·(1 − 2.25577×10⁻⁵·z)^5.25588 kPa, z in m above sea
    level;
  - the bulk-Ri scalar dtype fix.

  `physics/constants.py` holds the universal constants: `R`, `CP`, `R_CP`,
  `REFERENCE_PRESSURE = 100.0` kPa, `WATER_AIR_MWR = 0.622`, `STANDARD_GRAVITY`,
  `KAPPA = 0.41` and Earth's rotation rate. A formula's own coefficients (Magnus, ISA,
  Businger–Dyer) are named constants next to it. Functions specified in later phases
  (Yamartino, the autocovariance, the anisotropy port, …) are added to these modules in those
  phases.
- `config/` as in §2.2 and §3.8. The dataclasses mirror config-reference.md exactly: sections →
  nested frozen dataclasses; tables such as `min_mad` become `dict[str, float]`.
- `logs.py` (§3.9).
- `tests/conftest.py`: the `--integration` option (skip marked tests without it), a
  `raw_dir` fixture (§1.4), and an autouse, session-wide fixture that sets `TTU_TOWER_HOME` to a
  temporary directory, so no test ever touches the real ttu-tower home.

**Acceptance**
- `pip install -e ".[dev]"` succeeds in the 3.13 venv; `pytest` runs.
- The template loads. Every default in config-reference.md is what `load_config` produces for a
  minimal config (only the required keys).
- An unknown key, and a violation of each cross-key rule, each raise `ConfigError` naming the
  key.
- Round trip: `Config` → resolved JSON → `Config` gives an equal object.
- The adapted windprofiles tests pass (`test_polar`, the fit tests of `test_stats` including
  `test_power_fit_weighting_reduces_bias_vs_nls`, `test_atmos`, `test_geo`), and the bulk-Ri
  scalar dtype test.
- Magnus: e_s and the dewpoint match published table values, and e_s(T_d) = e to 1e-12.
- Hashes are stable under key reordering. Changing a `[secondary]` key changes the secondary
  and tertiary hashes but not the primary's. `nproc` changes no hash.

### Phase 1: Core data model

**Goal:** the time grids, the flag store, the table machinery and the results registry that
everything else builds on.

**`timegrid.py`**
- `EPOCH` and the conversions of §3.1: `time_to_sample`, `sample_to_time`, `slot_to_time`,
  `time_to_slot` (requires alignment), `half_hour_to_time`, `time_to_half_hour`.
- `BlockGrid(num: int, den: int)`, with `FLOOR`, `FINEST`, and a helper returning an
  integer grid.
- `block_sums(values: np.ndarray (n,) or (n, m), weights: np.ndarray (n,), g0: int, grid,
  b0: int, nb: int) -> tuple[np.ndarray (nb, m), np.ndarray (nb,)]`. Asserts that the samples
  fully cover blocks b0..b0+nb−1. NaN values must carry weight 0; the function zeroes x where
  the weight is 0, so NaN never propagates.
- `majority_block(g, grid)`.

**`flags.py`**
- `TESTS: dict[str, TestSpec]` (name, kind, grain, variables); `BOOM_LEVEL_VARIABLES = ("ue",
  "vn", "w", "ts")`.
- `mask_to_intervals(mask, g0) -> (starts, ends)` (runs of True, as `[start, end)`).
- `FlagRows`, an accumulator: `.add(test, boom, variable, starts, ends)` and `.frame()`, which
  merges touching or overlapping intervals per (test, boom, variable) and casts to the schema.
- `FlagStore(frame)`, which builds per-key sorted, merged interval arrays and prefix lengths:
  - `.count(test, boom, variable, starts, ends) -> np.ndarray[int64]`: flagged samples within
    each `[start, end)`, O(log n) per query. A query for a variable includes boom-level (null)
    rows when the variable is in `BOOM_LEVEL_VARIABLES`.
  - `.fraction(...)` = count / (end − start).
  - `.mask(tests, boom, variable, g0, n) -> bool[n]`.

**`schema.py` and `io/store.py`**
- One `TableSpec` per table in output-schema.md: ordered columns, dtypes, categorical columns.
  `cast(name, df)` validates (missing or extra columns are an error) and casts.
- `write_fragment(table_dir, key, df)` (atomic: write `…​.tmp`, then `os.replace`);
  `read_table(table_dir, columns=None, filter=None)`, which unifies the fragment schemas first
  (port the old `io/dataset.py` fix, where a null-only categorical column in one fragment must
  not drop values from others); `write_json_atomic`; manifest read/write.
- `TableSpec` also records the table's stage and whether it is written as fragments or as one
  file (`files`, `slots`, the monthly `wide` files), for `load_results`.

**`io/runs.py` and `results.py`**: exactly §3.10: `find_home`, `run_dir_for(cfg, test)`,
`register_run`, `find_run`, `load_results`, `list_runs`. Validate the tag against
`^[A-Za-z0-9._-]+$` at config load, since it names a directory and a file.

**Acceptance**
- Time conversions round-trip for random instants, including Feb 29 2012 and 2014-11-01.
- `block_sums`:
  - on an integer grid it equals reshape-and-sum;
  - on the floor and finest grids, with all weights 1, each block's weight is exactly
    num/den (to 1e-12);
  - the total over blocks equals the total over samples;
  - results for blocks inside a sub-span are bitwise identical to the full span's
    (translation invariance);
  - NaN with weight 0 doesn't propagate.
- `FlagStore`:
  - for random masks and random supports, `count` equals brute force;
  - boom-level rows are included for sonic variables only;
  - intervals split across two fragments give the same counts as the merged interval.
- Two fragments, one with an all-null categorical column, read back with no values lost.
- **Home discovery**, with `user_home` a temporary directory and `TTU_TOWER_HOME` unset:
  - nothing there → `.ttu-tower` is created, with the signature;
  - `.ttu-tower` exists without the signature → `.ttu-tower0` is created;
  - `.ttu-tower` foreign, `.ttu-tower0` absent, `.ttu-tower1` ours → `.ttu-tower1` is found and
    nothing is created;
  - a *file* named `.ttu-tower`, and a signature file with the wrong content, are both treated
    as foreign;
  - a second call finds the home the first one created.
- **`TTU_TOWER_HOME`:** adopted when it has the signature, initialized when missing or empty, an
  error when it is a non-empty foreign directory.
- **Registration:** all four cases of §3.10, including the stop message and `force`. Test runs
  are registered under `runs/testing/` and never collide with a normal run of the same tag.
- **Loading:** with fake fragments written through `register_run` (once with an external
  `results_dir`, once with the default), `load_results` returns them for a fragment table, a
  single-file table and `wide` (monthly files concatenated in order); it raises the documented
  errors for an unknown tag and a missing run directory; `list_runs` reports both runs.
- The suite's autouse fixture works: `find_home()` inside a test is not under the real user
  home.

### Phase 2: Stage A and Stage B operations (pure array functions)

**Goal:** every per-sample and windowed QC operation as a tested pure function on one boom's
arrays. No file I/O, no buffering. Each function takes the global index `g0` of its first
sample, so window grids anchor correctly.

**`primary/stage_a.py`**
- `to_si(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]` (§3.2).
- `apply_bounds(si, cfg.bounds) -> (si, removed: dict[str, bool mask])`. For the sonic triplet,
  a sample outside `bounds.sonic` in any component is removed from all three.
- `couple_triplet(u, v, w)`: NaN in any → NaN in all.
- `tilt_correct(u, v, w, boom) -> (u, v, w)`: exact port of `fix_sonic_tilt_misalignment`
  (R_total = Rnz·Ry·Rz, γ = `TILT_ANGLES[boom]`, θ = `TILT_AXES[boom]`, in degrees).
- `rotate_to_earth(u, v) -> (ue, vn)` = (v, −u).
- `stage_a(raw, boom, cfg) -> StageA` (dataclass: `ue, vn, w, ts, t, rh, p` float64, and the
  bounds removal masks).

**`primary/despike.py`**: `despike(x, g0, cfg_despike, min_mad, c) -> Despiked(x, spike,
excursion, unchecked)`.
1. Reference centers at global multiples of S = `stride_s`·50. For each center whose full
   window `[center − H, center + H)` (H = `window_s`·25) lies in the span:
   - valid if finite count ≥ c·2H;
   - median m and MAD = max(median(\|x − m\|), `min_mad`), over finite samples.

   Use `np.lib.stride_tricks.sliding_window_view` on the span, stepping by S, then
   `np.nanmedian` along the window axis.
2. Per sample, between the bracketing centers:
   - if both are valid, interpolate m and MAD linearly;
   - if exactly one is valid, use it;
   - if neither is valid, use the nearest valid center within H;
   - otherwise the sample is `unchecked`.
3. z = 0.6745·\|x − m\|/MAD; outlier = z > `z_threshold` (NaN is never an outlier).
4. Runs of consecutive outliers are split wherever x − m changes sign. Each piece of length
   ≤ `max_spike_samples` → `spike` (set to NaN); longer → `excursion` (kept).

The caller applies triplet coupling to the ue/vn/w spike masks (the union across the three is
removed from all three and recorded on all three).

**`primary/gaps.py`**
- `nan_runs(x) -> (starts, ends)`: interior and edge runs.
- `fill_short(x, max_gap) -> (x_filled, filled_mask)`: interior NaN runs of at most `max_gap`
  samples, with finite neighbours on both sides, are filled by linear interpolation between
  those neighbours.
- `fill_within_runs(x, file_run_id, max_gap)`: the same, restricted to gaps whose two
  neighbours share a `file_run_id` ≥ 0 (the same file run). Used for `mrd_unexcised`, with
  `max_gap` = `unexcised_max_fill_gap_s`·50.

**`primary/qc_windows.py`**
- `resolution_dropouts(x, g0, cfg_windows, c, is_ts) -> (resolution: list of intervals,
  dropouts: list of intervals)`. Windows start at global multiples of
  `resolution_stride_s`·50, have length `resolution_window_s`·50, and lie wholly in the span.
  Port the per-window logic of the old `_rdbin_series` (old `qc/tests.py`):
  - skip the window if finite coverage < c;
  - bin range [mean − 3.5σ, mean + 3.5σ] if that is narrower than [min, max], else [min, max];
    `resolution_bins` bins (`ts_resolution_bins` for ts);
  - resolution: flag the window if empty bins > int(bins × `max_empty_bin_fraction`);
  - dropouts (skipped in resolution-flagged windows): mark consecutive pairs that are both
    finite and in range and fall in the same bin (for ts: \|Δ\| ≤ `ts_repeat_tolerance`),
    keeping only runs of ≥ `dropout_min_run` marked pairs (`ts_dropout_min_run` for ts);
  - a pair is a dropout if marked in ≥ min(`dropout_min_windows`, number of evaluated windows
    containing it);
  - flag a window if its dropout-pair fraction (dropout pairs / (window length − 1)) exceeds
    `max_dropout_fraction`.

  Return the flagged windows' intervals.
- `higher_moments(x, g0, slots, cfg_windows, c) -> (skew, kurt, skew_flag, kurt_flag)`, per
  slot. Uses finite samples (after short filling) and requires coverage ≥ c. The statistics are
  in `math/stats.py`: `linear_detrend(x)` (least squares on sample index) and
  `moments(x) -> (skew, kurt)`, the population moments skew = m3/m2^1.5 and kurt = m4/m2². It
  flags `skew` outside `skew_range` and `kurt` outside `kurt_range` (values on the limits
  pass).

**`primary/slow.py`**
- `hann_kernel(width_s) -> k`: k_j = cos²(π·j/(2H)) for j = −(H−1)..H−1, with
  H = `width_s`·25 samples (half the width). The window is zero at ±`width_s`/2, so its
  reach is H − 1 samples. Default widths (`smoothing_width_s`, twice the sensor response
  time): t and rh 20 s (999 weights), p 2 s (99 weights).
- `smooth(x, ok, k, c) -> (xs, usable)`, a NaN-aware normalized convolution evaluated at every
  sample (design-reference E):
  1. x0 = x where `ok` (first-layer usable), else 0;
  2. S = `np.convolve(x0, k, "same")` and W = `np.convolve(ok.astype(float), k, "same")`;
  3. `usable` = W ≥ c·Σk (weighted coverage of the kernel); xs = S/W where usable, else NaN.

  Use the direct `np.convolve`, not an FFT: each output is then a fixed-order sum over its own
  window, so results are bitwise identical whatever the span (§3.7). Cost: ~30 ms per variable
  per file.
- `vpts(ts, boom) -> vpts` = ts·(100/`P_REF[boom]`)^R_CP, via
  `physics.thermo.potential_temperature`. The mask is ts's own; the measured-pressure
  correction is applied in secondary (Phase 7).

**`primary/masks.py`**
- `first_layer(finite: dict, intervals, unusable_tests, g0, n) -> dict[str, bool]` (§3.4).
- `family_masks(masks) -> {"momentum", "heat", "ts"}`.
- `second_layer(ue, vn, momentum_mask_l1, g0, slots, cfg_second, c) -> (wd_l1, ws_mean_l1,
  ws_std_l1, direction_flag, bounce_flag)`, per slot:
  - only if the momentum first-layer coverage ≥ c;
  - wd from the vector mean (`vector_to_bearing`); ws = mean and population std of per-sample
    √(ue² + vn²);
  - `direction` if low < wd < high (open interval); `bounce` if mean > `bounce_max_ws_mean` or
    std > `bounce_max_ws_std`.

**Regression fixture** (approved): a small script `tests/fixtures/make_stage_a_reference.py`,
run once with `old-tower-processing`'s own `.venv`. It loads the first 3,000 rows of
`FT2_E07_C03_R01457_D20131101_T0000.parquet` for every boom, applies the old unit conversion
(`windprofiles.processing.convert_dataframe_units`), `fix_sonic_tilt_misalignment` and
`compute_winds`, and saves the inputs and outputs to
`tests/fixtures/reference/stage_a_R01457.npz`. The script and the `.npz` are committed with
the tests; the script is never run by the test suite.

**Acceptance**

| Test | Setup | Expected |
|---|---|---|
| Stage A regression | `tests/fixtures/reference/stage_a_R01457.npz` | ue, vn, w, ts, t, rh, p match to 1e-10 |
| Wind direction end to end | adapted from the old `test_wind_direction_e2e.py`: raw (N, W) samples | mean wd equals an independently computed FROM-bearing to 1e-9 |
| Tilt | every boom | the matrix is orthonormal; boom 4's is the identity |
| Bounds and coupling | hand-built arrays | out of range → NaN; a sonic violation in one component removes all three |
| Spikes | Gaussian plus runs of 1–6 samples at 8σ | 1–3 → `spike` (NaN); 4–6 → `excursion` (kept) |
| Alternating burst | six consecutive outliers −10, 10, 8, −7, 15, −12 (× MAD, about a median of 0); a same-sign run 10, 14, 7, 12, 9, 11 | the first split into five pieces, all `spike`; the second one `excursion`, not split at its large jumps |
| Unchecked | a NaN stretch longer than a reference window | `unchecked` exactly where no valid reference lies within H |
| MAD floor | ts quantized at 0.01 K, near-constant | no mass flagging |
| Gap fill | gaps of 50 and 51 samples | the first filled, the second left |
| Within-run fill | a gap across a `file_run_id` change; a gap longer than the cap | neither filled |
| Resolution | u rounded to 0.5 m/s vs Gaussian | flagged vs not |
| Dropouts | 6% vs 4% stuck pairs in a window | flagged vs not |
| ts dropouts | exact repeats within `ts_repeat_tolerance` | flagged |
| Higher moments | log-normal σ = 0.7 (skewness ≈ 2.9); Gaussian scale mixture, 95% σ = 1 and 5% σ = 5 (kurtosis ≈ 20, symmetric); Gaussian | skew flagged; kurt flagged, skew not; neither |
| Smoothing kernel | `hann_kernel(20)`, `hann_kernel(2)` | symmetric, strictly positive, 999 and 99 weights; −3 dB point within 2% of 0.72/width |
| Smoothing values | a constant; a linear ramp; NaN runs within the kernel | constant and ramp unchanged away from gaps; correct renormalized mean next to NaN runs; weighted coverage < c → unusable |
| Smoothing has no steps | random walk with per-sample increments N(0, (0.001 K)²), rounded to 0.002 K; 20-s kernel | largest sample-to-sample output change < 0.15 × 0.002 K (a prototype gave ≤ 0.08) |
| Translation | despiking and smoothing of the same data with margin M vs 3M | bitwise identical core outputs |
| vpts | a ts series with the barometer entirely NaN | vpts = ts·(100/P_REF[b])^R_CP, usable wherever ts is; P_REF matches the ISA formula |
| Second layer | wd 104.9°, 105.1°, 170.0°; ws mean 30.1 m/s; ws std 6.8 m/s | none, direction, none; bounce; bounce |

### Phase 3: MRD engine

**Goal:** detection outputs from floor-grid sums, with the identities proven by tests.

**`primary/blocks.py`**
- `floor_sums(series, masks, g0, b0, nb) -> dict[family, FamilySums]`, where FamilySums holds
  `n` (nb,) and `sx` (nb, m):
  - momentum columns (ue, vn, w), weights = momentum mask;
  - heat columns (w, vpts − 273.15), weights = heat mask;
  - ts columns (ts − 273.15), weights = ts mask.

  The fixed offsets keep second-order sums well conditioned; means add the offset back.
- `finest_sums(series, masks, g0, b0, nb) -> dict[family, FamilySums]`, with first- and
  second-order columns:
  - momentum: ue, vn, w, ue², vn², w², ue·vn, ue·w, vn·w, ws, ws²; plus a direction sub-sum
    with its own weight n_dir (momentum mask ∧ ws > 0) over ue/ws and vn/ws;
  - heat: w, θ, w², θ², w·θ (θ = vpts − 273.15);
  - ts: τs, τs² (τs = ts − 273.15).
- Both are computed once per variant: `mrd` with the final masks, `mrd_unexcised` with the
  unexcised weights (§3.4).

**`primary/mrd.py`**
- `mrd_modes(sx, sy, n, block_len, c) -> (value[15], se[15], n_pairs[15])`, where `sx`, `sy`
  and `n` are the floor sums over one detection window (32,768 blocks) and block_len = 1875/256.
  For mode i = 1..15 (scale P_i = 2^i floor blocks; each block of width P_i is a pair of
  halves):
  1. aggregate the floor sums into halves of h_i = 2^(i−1) floor blocks: reshape to (−1, h_i)
     and sum `sx`, `sy`, `n`;
  2. half coverage = n_agg/(h_i·block_len); a half is valid if its coverage ≥ c;
     means = s/n_agg;
  3. each block = halves (2j, 2j+1). It is valid if both halves are, and its product is
     p_j = Δx·Δy/4 with Δ = mean(2j+1) − mean(2j);
  4. value = mean of p over valid blocks; se = std(p, ddof=1)/√n_valid if n_valid ≥ 2, else
     NaN; `n_pairs` = n_valid (valid blocks, each a pair of halves).
- `detection_outputs(floor: dict[family, FamilySums] for the window, c) -> (spectra: dict[str,
  (value, se, n)], wd_deg, coverage_momentum, coverage_heat)`:
  - φ from the window totals Σue, Σvn (momentum), and wd_deg = the FROM-bearing of that vector;
  - rotate the momentum sums: sx_u = cosφ·Σue + sinφ·Σvn, sx_v = −sinφ·Σue + cosφ·Σvn (exact,
    since rotation is linear);
  - spectra (mode products per family):

    | Spectrum | Variables | Family |
    |---|---|---|
    | `uw` | u·w | momentum |
    | `vw` | v·w | momentum |
    | `uv` | u·v | momentum |
    | `uu`, `vv`, `ww` | variances | momentum |
    | `wvpts` | w·θ | heat |
    | `tsts` | τs·τs | ts |

  - if the window has no momentum weight, momentum spectra and wd are NaN (likewise for the
    other families).

**Acceptance**
- **Identity** (fully valid window, 20 random seeds): for each rung τ_j = P_(j+6), the sum of
  `value` over modes i ≤ j + 6 equals the mean within-τ_j-block population covariance of the
  floor-block means (1e-10 relative). The sum over all 15 modes equals the covariance of the
  floor means about the window mean.
- **Floor invariance:** with an integer floor of 8 samples per block and 2^18 samples, modes at
  scales ≥ 8 samples computed from raw samples equal those from pre-averaged blocks exactly.
- **Valid-pair rule** (c = 0.8, so no coverage lands exactly on the threshold; one 75-s stretch,
  aligned to the 75-s grid, blanked):

  | Mode (scale) | Blocks if clean | Blocks expected |
  |---|---|---|
  | 1–9 (0.29 s … 75 s) | 2^(15−i) | 2^(15−i) − 256/2^(i−1) |
  | 10 (2.5 min) | 32 | 31 |
  | 11 (5 min) | 16 | 15 |
  | 12 (10 min) | 8 | 7 |
  | 13 (20 min) | 4 | 4 |
  | 14 (40 min) | 2 | 2 |
  | 15 (80 min) | 1 | 1 |

  At larger scales the stretch lies inside one half of one block, whose coverage is 50% at
  5 min (invalid), 75% at 10 min (invalid) and 87.5% at 20 min (valid). No value is ever
  computed from an invalid half.
- **White noise:** for i.i.d. N(0, 1) floor means over 200 windows, the mean of `value_i` is
  within 3 standard errors of 1/(2·2^(i−1)).
- **SE** equals std(ddof=1)/√n of the block products, recomputed by brute force.
- **Rotation:** input winds rotated by 37° leave `wvpts`, `ww` and `tsts` unchanged and rotate
  (`uw`, `vw`) as a vector; a westerly wind gives `wd_deg` = 270.

### Phase 4: Statistics ladder, integral timescales, transport efficiencies, slot statistics

**Goal:** everything primary reports per rung and per slot, as pure functions of block sums and
sample arrays.

**`primary/ladder.py`**: `rung_statistics(finest: dict[family, FamilySums] over the 128 finest
blocks [64k − 32, 64k + 96), c) -> (values: rows of (rung_s, variable, stat, value), coverage:
rows of (rung_s, family, blocks_used, blocks_total, coverage))`.

1. For j = 0..6, the slot's 64 finest blocks are grouped into 64/2^j τ-blocks. For j = 7 there
   is a single τ-block: all 128.
2. For each τ-block and family, aggregate the sums. Coverage = n/(2^j·468.75); the block is
   valid if coverage ≥ c.
3. For each valid momentum block:
   - means ū_e, v̄_n, w̄; the Earth-frame covariance matrix C_ab = Σab/n − ā·b̄;
   - rotate by φ_b (§3.3), giving `u/v/w var` and `uv/uw/vw cov`;
   - u mean = √(ū_e² + v̄_n²);
   - ws mean = Σws/n and ws var = Σws²/n − mean²;
   - Yamartino σ_θ (in degrees) from the direction sub-sums, s̄ = Σ(vn/ws)/n_dir and
     c̄ = Σ(ue/ws)/n_dir: `math.polar.yamartino_std(s̄, c̄)` computes ε = √max(0, 1 − s̄² − c̄²)
     and σ_θ = asin(ε)·(1 + (2/√3 − 1)·ε³).
4. For each valid heat block: `wvpts cov`, `vpts var`, `vpts mean` (+273.15). For each valid ts
   block: `ts var`, `ts mean`.
5. The rung value is the **unweighted mean over valid blocks** of each block value, except:
   `ws std` = √(mean var_ws), `wd std` = √(mean σ_θ²). If no block is valid, the value is NaN.
6. Coverage row per family: blocks_used, blocks_total, and coverage = Σn over the support /
   (samples in the support).

Clamp tiny negative variances from rounding to 0.

**`primary/rawrung.py`**: `rung_its_te(series, masks, g0, k, cfg_ladder, c) -> rows`. Inputs
are the final or unexcised series and masks for samples `[30000k − 15000, 30000k + 45000)`.

For each rung j, τ-blocks tile the slot (j ≤ 6) or form the 20-min window (j = 7). Samples are
assigned by the majority rule (§3.1).

- **ITS** (u, v, w, vpts):
  - a block is used only if every sample is usable in the family the variable belongs to —
    momentum for u, v and w, `ts` for vpts (for `mrd_unexcised`: finite) — and the two are
    counted separately (families `its` and `its_vpts`);
  - rotate u, v by the block's own mean direction (from its samples) and subtract the block
    means; vpts is not rotated, only centred on its block mean. Its ITS equals the ITS of ts,
    since they differ by a constant factor, and the slot's pressure factor is constant too, so
    secondary converts nothing;
  - autocovariance sums S_b(ℓ) = Σ_i x_i x_{i+ℓ} for lags ℓ = 0..L, with
    L = floor(`its_max_lag_fraction` × nominal block length): `math.stats.autocovariance(x,
    max_lag)`, by FFT with zero padding to the next power of two ≥ 2·len;
  - pooled ρ(ℓ) = Σ_b S_b(ℓ)/Σ_b S_b(0);
  - ITS = 0.02 s × `math.stats.efolding_integral(ρ)`: the trapezoid integral of ρ from lag 0 to
    the first crossing of 1/e, linearly interpolated between the bracketing lags; NaN if ρ never
    reaches 1/e by lag L (design-reference M).
  - Also record blocks_used/blocks_total (`its` for u, v, w; `its_vpts` for vpts).
- **TE:**
  - on the same valid blocks as the ladder (coverage ≥ c), over usable samples; fluctuations
    about the block's usable mean, with u rotated to the block's streamwise frame;
  - pool S_all = Σ u'w', S_pos = Σ max(u'w', 0), S_neg = Σ min(u'w', 0) over all valid blocks;
  - `physics.turbulence.transport_efficiency(S_all, S_pos, S_neg)`: TE = S_all/S_pos if
    S_all > 0, else S_all/S_neg; NaN if the denominator is 0;
  - the same for w'θ' in the heat family. Ported from the old `compute_transport_efficiencies`
    (Salesky et al. 2017), pooled over blocks.

**`primary/slotstats.py`**: `slot_means(series, masks, slow, g0, k, cfg, c) -> rows` for
`means`:
- momentum-family means of ue, vn and w;
- ws mean (scalar), ws vector_mean, wd mean (`vector_to_bearing` of the mean vector), and wd
  unit_mean (`vector_to_bearing` of the mean of the per-sample unit vectors (ue, vn)/ws, over
  usable samples with ws > 0);
- ws max, w max;
- for each p in `gust_periods_s`: blocks of 50p samples tiling the slot, valid if momentum
  coverage ≥ c; speed = magnitude of the block-mean vector; w = block mean; max and population
  std over the valid blocks (std NaN if fewer than 2);
- ts mean, vpts mean (their own masks);
- t, rh, p means of the smoothed series on their own masks.

**Acceptance**
- **Aggregation = direct computation**, for every rung including the fractional 9.375-s and
  18.75-s ones, on 20 random slots with random masks: each τ-block's weighted means and
  covariances computed directly from samples (same fractional weights), rotated per block and
  averaged, equal `rung_statistics` to 1e-10 relative. For rungs ≥ 37.5 s, also check against
  plain numpy on integer slices.
- **Rotation:** for synthetic flow with a known mean direction and fluctuations aligned with it,
  `v var` ≈ the injected value and `uv cov` ≈ 0. Rotating all Earth-frame input by a constant
  angle leaves every ladder output unchanged (1e-12).
- **Yamartino** equals the single-pass formula applied directly to the samples, and approaches
  the circular standard deviation for small spreads.
- **Coverage:** blanking one τ-block drops exactly that block (`blocks_used` − 1), and the value
  is the mean of the rest.
- **ITS of vpts** equals the ITS of ts to 1e-12 on the same data, and uses the `ts` family's
  blocks: blanking a sample that only the momentum mask covers changes the u/v/w ITS blocks and
  not the vpts ones.
- **Wind direction means:** with constant speed, `unit_mean` equals `mean`; with a slow sample
  at one bearing and fast samples at another, `mean` follows the fast samples while `unit_mean`
  sits near the middle; both match an independent computation to 1e-9.
- **ITS** on an Ornstein–Uhlenbeck process (T = 5 s). The e-folding integral of an exponential
  ACF is (1 − e⁻¹)·T ≈ 3.16 s:
  - at the 20-min rung, the mean over 20 realizations is within 10% of it;
  - at the 9.375-s rung (maximum lag 2.3 s) the result is biased low or NaN; record which
    (expected, not a failure);
  - a constructed ACF that never reaches 1/e gives NaN.
- **TE:** 1 when all products share a sign; a constructed 3:1 split gives the analytic value.
- **Gusts:** a step signal gives the known block maxima; an invalid block is skipped.

### Phase 5: Raw data layer

**Goal:** the file table (timing rule) and per-boom loading, validated on the real listing.

**`io/rawfiles.py`**
- `parse_name(name) -> (record, name_time, kind) | None`, with the regex
  `^FT2_E07_C03_R(\d+)_D(\d{8})_T(\d{4})\.(parquet|csv|csv\.gz|zip)$` (time in `Etc/GMT+6`,
  the source's zone, whatever the output zone). `kind` is `parquet` or `raw`. Port the old
  `get_datetime_from_filename` semantics.
- `check_raw_allowed(table, allow_non_parquet)`: if the table has any `raw` files and the flag
  isn't set, stop with the message in §2.3 (the count and directory, what to run, and the
  flag). With the flag, log a warning with the count and continue.
- `build_file_table(raw_dirs, cfg_files) -> DataFrame` (output-schema.md §3.1), checked in this
  order:
  1. `unparseable`;
  2. `bad_record` (record in any inclusive range);
  3. `offset` (minutes past the preceding :00/:30 > `max_name_offset_min`; rounded start =
     that boundary);
  4. `collision` (two or more surviving files with the same `half_hour`: all of them
     rejected, and a warning logged);
  5. `bad_length` (Parquet metadata row count ≠ 90,000; not checked for raw files, whose length
     `load_boom` checks instead);
  6. otherwise `accepted`.
- `accepted_half_hours(table) -> dict[int, path]`.

**`io/load.py`**: `load_boom(path, boom, temp_dir) -> dict[str, np.ndarray]`.
- For parquet, it reads only `u_b, v_b, w_b, ts_b, t_b, rh_b, p_b` with
  `pyarrow.parquet.read_table(columns=…)`.
- For a raw file (only reachable with `--allow-non-parquet`), it runs `convert.convert_raw_file`
  in memory and takes the same columns. That parses the whole CSV for each boom, which is why
  it's for small runs.
- Returns float64 arrays. Missing columns raise.
- A length other than 90,000 raises `BadFileError`. The stream treats that file as missing (a
  placeholder) and logs a warning; this is only possible for raw files.

**`io/convert.py` and `cli/convert_parquet.py`**: port the old `io/raw.py` conversion pieces
(`_resolve_zip`, `_read_raw_csv`, `infer_source_schema`, `convert_raw_file`) and the whole old
`cli/convert_parquet.py`, dropping the `windprofiles` dependency. `rename_headers` becomes a
simple dict-driven rename using `HEADER_MAP_OLD/NEW`, where a `None` target drops the column.
The output must be identical to the existing converted files: same columns, float32.

**`cli/files.py`**: prints counts by status and the coverage of the period's half-hours, and
writes `primary/files.parquet`.

**Acceptance**
- Parsing: valid names, malformed names, and names at 23:5x crossing midnight.
- Rule order on a synthetic listing: a clock-anomaly record inside `bad_records` is dropped
  before rounding could collide it; offsets +2 are accepted and +3 rejected; two files rounding
  to one half-hour are both rejected.
- **Real listing (integration):** 16,817 files → 1,686 `bad_record`, 133 `offset`,
  0 `collision`, 0 `bad_length`, 0 `unparseable`, **14,998 accepted**. Over the period
  2013-11-01 00:00 → 2014-11-01 00:00 that is 14,998/17,520 = 85.6% of half-hours.
- The converter reproduces the existing parquet for the old repo's fixture
  (`old-tower-processing/tests/fixtures/raw/FT2_E07_C03_R01457_D20131101_T0000.zip`, copied into
  `tests/fixtures/raw/`): same column names and dtype, equal values.
- `load_boom` returns seven float64 arrays of length 90,000 for a real file. For the fixture
  raw zip (with `--allow-non-parquet`), it returns the same values as for the converted parquet.
- Without the flag, a directory containing one `.zip` stops `ttu-files` with the §2.3 message.
- **Quantization survey (integration, read-only; report to Elliott):** on 50 accepted files
  spread over the year, for every boom and variable, report the smallest nonzero step between
  distinct values, and the fraction of 5-min windows whose MAD is below `min_mad`. This checks
  the `min_mad` defaults (design-reference E); the numbers are reported, not asserted.

### Phase 6: Rolling buffer and primary orchestration

**Goal:** primary end to end: batches, streaming, Stage B composition, products, parallel
units, checkpoints, reports. Seam invariance and parallel = serial are proven by tests.

**`primary/partition.py`**: `plan_batches(file_table, period, batch_max_files) -> (batches:
list[Batch], outage_half_hours: list[int])`.
- The period's half-hours are those overlapping `[period.start, period.end)`.
- **Outages:** runs of ≥ 2 consecutive non-accepted half-hours. They belong to no batch, and
  their slots are emitted as `no_file` by the runner.
- The stretches between outages (which may contain isolated missing half-hours) are split into
  consecutive batches of at most `batch_max_files` half-hours.
- `Batch(h_a, h_b)` with `.id` = `bat{h_a:07d}-{h_b:07d}`. Its context: Stage A for
  [h_a − 3, h_b + 3], Stage B for [h_a − 2, h_b + 2], products for [h_a, h_b]. Accepted files
  are used even outside the period (design-reference K); non-accepted half-hours are
  placeholders. Only slots inside the period are emitted.

**`primary/stage_b.py`**: `stage_b(span: Span, h: int, boom: int, cfg) -> StageBOut`.
- `Span` holds the Stage A arrays for `[90000h − M, 90000(h+1) + M)` (M = 30,000 samples, the
  10-min margin), assembled from the neighbours or NaN. It also holds `file_run_id` per
  sample: −1 for placeholder samples, otherwise the half-hour index of the first file of that
  contiguous accepted run.
- Order:
  1. despike each variable (§ Phase 2), with triplet coupling of ue/vn/w spike removals;
  2. `fill_short`;
  3. `higher_moments` (core slots);
  4. `resolution_dropouts` (ue, vn, w, ts);
  5. `smooth` (t, rh, p);
  6. `vpts`;
  7. first-layer masks;
  8. `second_layer` (core slots);
  9. final masks;
  10. the unexcised series (`fill_within_runs` on the Stage A-plus-despiked series, then vpts
      from it) and its weights;
  11. `floor_sums` and `finest_sums` for both variants, over the core's 12,288 floor and
      192 finest blocks;
  12. per slot: coverage rows, `slot_means`, `slot_qc` rows, `mask_differs` (any core sample of
      ue/vn/w/ts/vpts finite in the unexcised series but not usable in the final mask).
- Flag rows are clipped to the core.
- Keep only core results. `StageBOut` also carries the core's final and unexcised series and
  masks, for rawrung and slot statistics.
- A placeholder `StageBOut` for a missing file has NaN series, zero weights and no rows.

**`primary/products.py`**: `file_products(h, B: Mapping[int, StageBOut], boom, cfg) -> dict[table,
DataFrame]` for the three slots k of file h.
- **No file:** only a `slot_boom` row with status `no_file`.
- **`no_data`:** no final-usable ue/vn/w/ts sample in the slot. The `slot_boom`, coverage and
  `means` rows (the slow-sensor means stay valid when the sonic data are flagged; the sonic means
  are NaN), and no `mrd` products. A slot can be `no_data` because everything in it is flagged
  (e.g. by the tower wake) while data exist. Then, if `primary.unexcised` is set and the
  unexcised series has data in the slot, compute the `mrd_unexcised` products (below) and set
  `unexcised_computed = true`: that is exactly the counterfactual the variant exists for.
- **Computed:** coverage, means, slot_qc; `mrd` and `mrd_frame` from `detection_outputs` on the
  window's floor sums assembled from B[h−2..h+2]; `ladder` and `ladder_coverage` from
  `rung_statistics` on the finest sums from B[h−1..h+1] and `rung_its_te` on the series from
  B[h−1..h+1]. Do this for `mrd`, and for `mrd_unexcised` iff `primary.unexcised` and
  `mask_differs` is true in any of slots k−4..k+4. Set `unexcised_computed` accordingly.
- Flags: B[h]'s rows, written once per file.

**`primary/stream.py`**: `run_unit(batch, boom, file_table, cfg, out_dir) -> UnitSummary`.

```
for h in h_a−3 .. h_b+3:
    A[h] = stage_a(load_boom(path, boom)) if h is accepted else PLACEHOLDER
    hb = h − 1                     # Stage B for hb needs A[hb−1..hb+1]
    if h_a−2 ≤ hb ≤ h_b+2: B[hb] = stage_b(span(A, hb), hb, boom, cfg)
    hp = hb − 2                    # products for hp need B[hp−2..hp+2]
    if h_a ≤ hp ≤ h_b: collect(file_products(hp, B, boom, cfg))
    drop A and B entries no longer needed
write all collected tables as fragments <unit>.parquet (atomic), then the manifest entry
```

(At the top of the range, h_b+3 is loaded only as Stage B's right margin.)

**`primary/runner.py`**: `run_primary(cfg, args) -> RunSummary`.
1. Register the run directory (§3.10), then the hash guard (§3.8). Secondary and tertiary
   start the same way.
2. Build or refresh `files.parquet`.
3. `plan_batches`.
4. Write `slots.parquet`, plus a `slot_boom` fragment `outages` holding `no_file` rows for
   every outage slot × boom.
5. List units (batch × boom); skip `success` ones in the manifest (and `failed` ones unless
   `--redo-failures`).
6. Dispatch to a `spawn` `ProcessPoolExecutor(nproc)` with the logging queue.
7. Write `run_summary.json` and print the console summary (§3.9).

`--test`: only the first batch. The manifest entry per unit is `{unit, status, error, started,
finished, summary}`, where `summary` is the unit summary record of §3.9.

**`primary/report.py` and `cli/report.py`** (primary part), writing CSVs to
`<run dir>/reports/`:
- `primary_files.csv`: file status counts, overall and by month.
- `primary_slots.csv`: slot_boom status counts per boom, by month.
- `primary_flags.csv`: per boom × variable × test, flagged-sample fraction of present samples,
  overall and by month.
- `primary_coverage.csv`: per boom × variable × layer: mean, 10/50/90% quantiles, and the
  fraction of computed slots ≥ c.
- `primary_second_layer.csv`: `direction` and `bounce` slot fractions per boom.

These are the "QC yield reporting early" deliverable. The stability breakdown comes in Phase 9.

**Acceptance** on synthetic data from `write_raw_dataset` (validation.md §2.4): 12 half-hours,
2 booms, with one isolated missing file, one 2-file outage, one file with boom 2 all-NaN, and
spikes, stuck runs and a long gap straddling file boundaries.

| Test | Expected |
|---|---|
| Seam invariance | the streaming outputs equal a reference that runs each file's Stage B with the whole file run as its span, then the same products: floats to 1e-12; flags, masks and statuses exactly |
| Batch partition | `batch_max_files` = 2 and 96 give identical outputs |
| Parallel = serial | `nproc` = 1 and 3 give identical outputs |
| Resume | a unit made to fail once completes after `--redo-failures`; the outputs equal a clean run's, with no duplicate fragments |
| Statuses | every slot × boom of the period appears once in `slot_boom`: outage and isolated missing file → `no_file`; the all-NaN boom → `no_data` |
| Flagged-only slot | a slot entirely inside a `direction` flag: `no_data`, `unexcised_computed` true, `mrd_unexcised` products present, no `mrd` products, slow-sensor means present |
| Unexcised trigger | `unexcised_computed` exactly for the slots within ±4 of a mask difference |
| Context | for a slot next to the isolated missing file, the detection window's floor sums have weight in the file beyond the gap, and fine-scale modes include blocks there; no Stage B output (fills, smoothing, despiking references) bridges the gap |
| Summaries | each manifest summary's slot counts equal that unit's `slot_boom` rows; `run_summary.json` totals equal the sums over units; the all-NaN boom's numpy warnings appear as counts, not log records |

**Real data** (integration; approved, and it writes results): 2013-11-01 00:00 →
2013-11-04 00:00, all booms, as a `--test` run. It completes with no failed units, every slot is
emitted, and the means are physically plausible. Report seconds per file-boom for Stage A,
Stage B, detection outputs, ladder and rawrung (from the unit summaries), extrapolated to one
year at the local core count (§6).

### Phase 7: Secondary

**Goal:** τ detection and selection, the selected-τ statistics and every single-boom derived
quantity, per variant.

**`secondary/detect.py`**: `detect(D, SE, N, family_has_data, cfg_det) -> Detection(status,
tau_s, tau_lb_s, sign, peak_scale_s, reversal_scale_s, reversal_type)` for one cospectrum (heat
= `wvpts`, momentum = `uw`) of one (slot, boom, variant): VM06's rule plus a significance test
on the peak (design-reference H). `family_has_data` means the family's coverage in the slot is
above zero (`coverage` layer `usable`, or `unexcised` for `mrd_unexcised`). Loop over the modes;
vectorize over rows where convenient.

Notation:
- P_i are the mode scales (§3.1); a reversal at mode r gives τ = P_(r−1);
- k = `peak_significance_se`;
- i_lo = the first mode with P_i ≥ `min_scale_s` (mode 2, 0.59 s, by default);
- i_top = the mode with P = 2·`max_tau_s`, which confirms `max_tau_s` (mode 14, 40 min);
- mode i is usable if N_i ≥ 2 and SE_i is finite (a standard error needs two blocks).

Steps:
1. If not `family_has_data`: `no_data`.
2. **Range.** u = the first mode in i_lo..i_top that isn't usable (u = i_top + 1 if all are).
   The analysed range is R = i_lo..u−1. Modes outside R are never smoothed or scanned.
3. **Smoothing (1–2–1).** For i in R, with weights (1, 2, 1) on modes (i−1, i, i+1), using only
   the modes in R:
   - D̃_i = Σ w_j·D_j / Σ w_j: (D_(i−1) + 2D_i + D_(i+1))/4 inside R, and e.g.
     (2D_i + D_(i+1))/3 at an end of R;
   - S̃_i = √(Σ w_j²·SE_j²) / Σ w_j, treating modes as independent.
4. **Peak.** Mode m ∈ R is a local maximum if m + 1 ∈ R and either sign(D̃_(m+1)) ≠ sign(D̃_m)
   or |D̃_(m+1)| < |D̃_m|. It is significant if D̃_m ≠ 0 and |D̃_m| ≥ k·S̃_m. The peak p is the
   first significant local maximum, scanning up from i_lo. If there is none:
   - if u ≤ i_top (R ended early): `unresolved`, τ_lb = P_(u−1);
   - else, if mode i_top is significant (the cospectrum still rises at the top): p = i_top;
   - else: `weak`.

   σ = sign(D̃_p).
5. **Gap.** The reversal r is the first mode after p in R with sign(D̃_r) ≠ σ (type `sign`) or
   |D̃_r| > |D̃_(r−1)| (type `increase`): `found`, τ = max(P_(r−1), `min_tau_s`).
6. **No reversal** through the end of R: `unresolved` with τ_lb = P_(u−1) if u ≤ i_top, else
   `capped` with τ = `max_tau_s`.
7. **Diagnostic for `capped`:** look at mode i_top + 1 (the 80-min mode: one block, unsmoothed)
   if its block is valid. It shows a `sign` reversal if its sign differs from σ, and an
   `increase` reversal if p < i_top and its magnitude exceeds |D̃_(i_top)|. If either, set
   `reversal_scale_s` to its scale and `reversal_type`; the status stays `capped`.

Record `sign` = σ (0 if no peak was found) and `peak_scale_s` = P_p; for `found`, also
`reversal_scale_s` = P_r and `reversal_type`.

**`secondary/select.py`**: `select(heat: Detection, momentum: Detection, cfg_sel) -> (tau_s,
source, source_status)`.
- `priority` rule:
  1. the first source in `priority` whose status is `found` or `capped` → (its τ, source,
     status);
  2. else the first `unresolved` source → (max(τ_lb, `fallback_tau_s`), source, `unresolved`);
  3. else, if either family has data → (`fallback_tau_s`, `fallback`, `fallback`);
  4. else (NaN, `none`, `none`).
- `longest_significant` rule: the largest τ among `found`/`capped` sources (ties by `priority`);
  otherwise steps 2–4.
- `naive` rows: (600, `fixed`, `fixed`).

Secondary writes rows for each (slot, boom, variant) that has primary ladder rows (including
the `mrd_unexcised` copies below); `naive` rows exist wherever `mrd` rows do.

**`secondary/derived.py`**
- `selected_stats(ladder, ladder_coverage, tau_selected, p_factor) -> rows`: the ladder rows at
  each row's selected rung, with the heat-family rows converted by the slot's pressure factor
  (below), plus the support coverage rows (`<family>` × `coverage`, `blocks_used`,
  `blocks_total`).
- `derived(stats, slow, heights, g, cfg) -> rows`: every derived quantity in output-schema.md
  §4.3, with zero denominators → NaN, using `physics.most` (u*, L, z/L) and
  `physics.turbulence` (TI, TKE, ILS, anisotropy). Anisotropy is ported from the old
  `tertiary_ops.py` (`_in_triangle`, `classify_anisotropy_states` with k = 2/3,
  `compute_anisotropy_angles`, `anisotropy_calculations`) into `physics/turbulence.py`, and
  applied to the selected-τ tensor; its class goes to `boom_labels`.
- **`mrd_unexcised` rows:**
  - where `unexcised_computed` is true: detection runs on the unexcised spectra, with
    `family_has_data` from the `unexcised` coverage layer;
  - where it is false and the slot is `computed`: copy the `mrd` detection, ladder and coverage
    rows under the `mrd_unexcised` label (they would be identical);
  - where it is false and the slot is `no_data`: no rows.

**`secondary/slow.py`**: the `slow` table of output-schema.md §4.5 from the `means` t, rh and p,
with the formulas listed there (`physics.thermo`), for every slot and boom with `means` rows. It
also gives the **pressure factor** (design-reference I):
- if the slot's `coverage` (`p`, `usable`) ≥ c: p̄ = `means` (`p`, `mean`),
  f = (P_REF[b]/p̄)^R_CP and `p_measured` = 1;
- otherwise f = 1 and `p_measured` = 0 (the reference pressure is kept).

`selected_stats` applies f to every variant and τ: `wvpts` cov and `vpts` mean × f, `vpts` var
× f²; `wvpts` te and every other row are unchanged. It does so before any derived quantity is
computed, so `obukhov_length` and `zeta` use the converted flux. Primary's own tables stay at
P_REF.

**`secondary/sun.py`**: sun elevation (astral `elevation`) and night (sunrise/sunset per UTC
date, cached per date as in the old `mark_night`), both at the slot center, for the observer
built from the site constants.

**`secondary/runner.py`**: batches in sequence, in one process. Per batch, all booms: read that
batch's primary fragments, compute, write fragments, log the batch summary (§3.9). Hash guard;
`run_summary.json` at the end. No checkpointing beyond per-batch fragments.

**Acceptance**
- **Detection cases** (SE = 0.05 unless stated; N_i = 2^(15−i); defaults k = 2 and
  `min_scale_s` = 0.5, so modes 2–14 are analysed). Each case is an explicit array in the test;
  the expected results come from a prototype of this algorithm. The smoothing moves reversals
  relative to the raw arrays, so the cases use smooth shapes unless the smoothing is under test.

  | Case | D (modes 1…15) | Expected |
  |---|---|---|
  | sign-type gap | 0.2, 0.5, 0.9, 1.3, 1.6, 1.7, 1.5, 1.0, 0.4, −0.3, −0.6, −0.4, −0.2, 0.3, 0.5 | `found`; peak 9.375 s; reversal at mode 10 (150 s, `sign`); τ = 75 s |
  | increase-type gap | 0.2, 0.5, 0.9, 1.3, 1.6, 1.7, 1.4, 0.9, 0.5, 0.3, 0.4, 0.8, 1.4, 2.0, 2.5 | `found`; peak 9.375 s; reversal at mode 11 (300 s, `increase`); τ = 150 s |
  | wobble before the peak (SE 0.1) | 0.1, 0.5, 1.0, 0.9, 1.6, 1.8, 1.5, 1.0, 0.4, −0.2, −0.4, −0.2, 0.1, 0.2, 0.1 | `found`; peak 9.375 s; reversal at mode 10 (`sign`); τ = 75 s. (Unsmoothed, the dip at mode 4 would end the peak at mode 3 and give 9.375 s, clipped from 2.34 s.) |
  | insignificant bump first | 0.0, 0.02, 0.06, 0.02, 0.0, 0.3, 0.8, 1.5, 2.0, 1.5, 0.6, −0.2, −0.3, −0.1, 0.1 | the local maximum at mode 3 (smoothed 0.04 < 2 × 0.031) is passed over; peak at mode 9 (75 s); reversal at mode 12 (`sign`); τ = 300 s |
  | mode 1 ignored | the sign-type array with D_1 = −5.0 | the sign-type result |
  | weak | 0.05, −0.04, 0.08, 0.02, 0.09, −0.05, 0.03, 0, 0.07, −0.02, 0.01, 0.05, −0.03, 0.02, 0.01 | `weak` |
  | rising through 40 min | 15 values evenly spaced from 0.1 to 3.0 | `capped`, τ = 1200 s; no diagnostic (mode 15 still rising) |
  | rising, 80-min sign change | 14 values evenly spaced from 0.1 to 2.8, then −1.0 | `capped`; diagnostic 4800 s, `sign` |
  | peak, slow decline, 80-min rise | 0.2, 0.5, 0.9, 1.3, 1.6, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 1.5 | `capped` (peak 9.375 s, no reversal through 40 min); diagnostic 4800 s, `increase` |
  | unusable mode | the rising array with N_12 = 1 | `unresolved`, τ_lb = 300 s |
  | gap below an unusable mode | the sign-type array with N_12 = 1 | `found`, τ = 75 s |
  | negative flux | −1 × the sign-type array | the same τ; sign −1 |
  | early reversal | 0.5, 1.5, 2.0, 1.0, −1.5, −0.3, 0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.2, 0.1, 0.1 | peak at mode 2 (0.59 s); reversal at mode 5 (`sign`); τ = 9.375 s (clipped from 2.34 s) |
  | SE = 0 | the sign-type array with SE = 0 | the same as with SE = 0.05 |
  | all zero, SE = 0 | D = 0, SE = 0 | `weak` |
  | no usable mode | the sign-type array with N_2 = 1 | `unresolved`, τ_lb = 0.29 s |
  | no data | a family without data | `no_data` |

- **Selection:** every combination of {`found`, `capped`, `unresolved`, `weak`, `no_data`} for
  (heat, momentum), under `priority` = [heat, momentum], [momentum, heat] and
  `longest_significant`, with the expected (τ, source, source_status).
- **Time-series detection** (seeds 0–49, `turbulence_with_waves` of validation.md §2.2): each
  realization is 3 h at 50 Hz with `ue` = 8 m/s + u, `vn` = 0 and `w`. Build the floor sums of
  the central slot's detection window, compute the modes, run detection on the momentum
  cospectrum, and take `cov_uw` from the ladder at the selected rung. Ratios are to the
  preset's analytic turbulent w'u'. A prototype of this algorithm gave:

  | Scenario (preset) | Statuses; selected τ | Mean ratio at the selected τ | Same-τ agreement | 600-s rung, for information |
  |---|---|---|---|---|
  | A: `MESOSCALE_STRESS` (opposing) | 50 `found`; 75, 150, 300 s in 4, 43, 3 | 0.985 | 50/50 | 0.938 |
  | B: `MESOSCALE_TYPICAL` | 48 `found`, 2 `capped`; median 300 s | 0.997 | 50/50 | 0.996 |
  | C: `STABLE_STRESS` (opposing) | 50 `found`; 9.375 s in all, none clipped | 0.868 | 48/50 | −0.344 |
  | D: `STABLE_TYPICAL` | 48 `found`, 2 `capped`; median 18.75 s | 0.978 | 40/50 | 0.997 |
  | E: C plus a 30-s wave | 50 `found`; 9.375 s in all, 35 clipped | 0.805 | 19/50 | −0.367 |
  | F: D plus a 30-s wave | 48 `found`, 2 `capped`; median 37.5 s | 1.002 | 34/50 | 1.004 |
  | G: A with `coupling` = `aligned` | 50 `found`; 75, 150, 300 s in 7, 37, 6 | 0.995 | 49/50 | 1.077 |
  | H: C with `coupling` = `aligned` | 50 `found`; 9.375 s in all, 1 clipped | 0.955 | 46/50 | 2.270 |

  "Same-τ agreement" counts realizations whose selected-τ flux is within 10% of the
  turbulence-only flux at the same τ; "clipped" means the reversal lay at or below 9.375 s, so
  τ was raised to `min_tau_s`. These tests check that the implementation behaves like the
  prototype (validation.md §1); they don't measure how well the algorithm suits real data. A
  scenario passes if its mean ratio is within 0.06 of the prototype's, its median selected τ is
  within one rung, and its `found`, clipped and same-τ counts are each within 7. The
  implementation's generator draws different random numbers, so the agreement is statistical. If
  it misses, report the distributions; don't retune silently.
- **Derived:** ti, tke, ustar, L, zeta, ILS (including `ils_vpts`) and anisotropy against hand
  calculations for three rows; zero denominators → NaN; naive rows equal the 600-s `mrd` rungs;
  the `mrd_unexcised` copy equals `mrd`.
- **`its_short`:** set by a low ratio in any of u, v, w, and by a variable with usable blocks
  whose ITS is NaN; 0.0 when all three are fine; NaN only when none had a usable block.
  `its_short_vpts` behaves the same way and is unaffected by the velocity components.
- **Pressure factor:**
  - vpts computed per sample at the slot's p̄, with the rung statistics then computed directly,
    equals the converted `boom_stats` values (1e-10), for τ = 600 s and τ = 1200 s;
  - with the slot's barometer coverage below c: f = 1, `p_measured` = 0, values equal
    primary's;
  - `wvpts` te is unchanged, and `obukhov_length` equals a hand calculation with the converted
    flux.
- **Slow thermodynamics:** the `slow` table equals `physics.thermo` applied by hand to three
  rows.
- **End to end on synthetic raw files**, which catches sign, unit, frame and offset errors
  between the stages: `write_raw_dataset` (validation.md §2.4) with 6 half-hours, 2 booms,
  `MESOSCALE_TYPICAL`, a westerly mean wind of 8 m/s and no defects, run through primary and
  secondary with `z_threshold` = 50, so that QC leaves the data untouched. For the four central
  slots of each boom, against a direct computation from `truth`:
  - the `naive` rows of `boom_stats`, to 1e-5 relative: u, v and w variances and uw, vw and uv
    covariances in the slot's streamwise frame; ts variance; the `wvpts` covariance as
    cov(w, ts)·(100/p̄)^R_CP, with p̄ from `means`; ustar and obukhov_length from these;
  - the slot means of ws and wd (FROM-bearing, independently computed);
  - `mrd` rows exist for every slot, with a `found` or `capped` source.

### Phase 8: Tertiary

**Goal:** mesonet merge, boom-level filtering, multi-boom quantities, profile fits, the wide
export.

**`tertiary/mesonet.py`**: port the old `mesonet.py` (`wtxmeso` reader: `stationinfo.xls`,
`data/`) and `processing/mesonet_merge.py`, including its gap handling
(`_fill_resampling_gaps`, `_extrapolate_edges`), with `avg_period = "10min"`. Output columns as
output-schema.md §5.5 (`ws_meso_rms` → `ws_meso_std`, `wd_meso_rms` → `wd_meso_std`). Join on
`slot_start` with both sides tz-aware, so the join works in either output zone. Import `wtxmeso`
lazily, only when
`mesonet_dir` is set, with a clear error if the `mesonet` extra isn't installed.

**`tertiary/filtering.py`**
- `QUANTITY_GROUPS`: a mapping (variable, stat) → group, and group → (input variables, required
  families), per output-schema.md §5.6. Every row of `boom_stats`, `means` and `slow` must map
  to exactly one group; a test enforces this.
- `evaluate(batch tables, FlagStore, cfg) -> fail mask + filter_log rows`:
  - supports: the slot `[30000k, 30000(k+1))`, and the 20-min window
    `[30000k − 15000, 30000k + 45000)` for variant rows whose selected τ is 1200 s;
  - per input variable: `bounds` fraction > `max_bounds_fraction` or `spike` fraction >
    `max_spike_fraction` → fail;
  - coverage: every required family's coverage ≥ `min_coverage`, taken from
    `ladder_coverage` at the selected rung for variant rows, or `coverage.usable` for `none`
    rows (for the vpt group, all of t, rh, p). The timescales are filtered on their family's
    sample coverage like everything else; their own block counts (`its`, `its_vpts`) are
    reported, not thresholded (output-schema.md §5.6).
- `apply(tables, fails)` sets failing values to NaN. Keep the criteria as a list of functions,
  each with a flag `kind` (all `quality` now), so that an assumption-kind criterion can be added
  later (design-reference L).
- The FlagStore for a batch is built from the flag fragments overlapping
  `[batch start − 5 min, batch end + 5 min)`, including the neighbouring batches' fragments.

**`tertiary/multiboom.py`** (on filtered `none` values):
- `rib` for all pairs i < j: `physics.richardson.bulk_richardson_number` with components = true
  (ue/vn means), slow VPT and local g.
- `lapse_vpt`.
- `veer` per boom vs `veer_reference_boom` (`math.polar.series_signed_angular_distance`: positive
  when the boom's wd is clockwise of the reference's).

**`tertiary/profiles.py`**
- Fits: weighted `power_fit` (y²-weighted in log space) from `math.fits`;
  `neutral_loglaw_fit` and `constrained_neutral_loglaw_fit` from `physics.most`.
- `alpha` (ws mean); `gamma` (ti) and `wdgamma` ((wd, std)) per variant; `loglaw_*`.
- Implement exactly the five profile-fit rules of output-schema.md §5.4, each driven by
  `[tertiary.fits]`, in one function per fit that takes the boom list, the τ-policy mask and the
  values. The rules must be readable in one place. Record `<fit>_n` and `<fit>_n_capped`.

**`tertiary/wide.py`**: the pivot of output-schema.md §5.7, one calendar month at a time (in
`[output].timezone`): read that month's rows of the tertiary tables, pivot, write
`wide/<YYYY-MM>.parquet`. A whole year pivoted at once would need about 1 GB of memory.

**`tertiary/runner.py`**: batches in sequence, in one process, like secondary. Mesonet is loaded
once and sliced per batch. After all batches, write the wide export month by month.
`cli/runall.py` runs primary → secondary → tertiary as subprocesses (port the old `runall`).

**Acceptance**
- The ported mesonet tests pass (adapted from the old
  `tests/unit/processing/test_mesonet_merge.py`).
- **Filtering:**
  - a bad-ts slot (bounds fraction 5%) NaNs the heat and ts groups but not momentum;
  - a spike fraction just above or below `max_spike_fraction` flips the decision;
  - a 20-min-τ row uses the 20-min support (a flag in the neighbouring slot's half counts);
  - low `ladder_coverage` NaNs only that variant's rows.
- **NaN propagation:** a filtered boom gives NaN `rib` for every pair containing it, and is
  excluded from fits (`_n` drops).
- **Profile policy:** `unresolved`/`fallback` booms are excluded from `mrd` fits but not from
  `naive` fits; `capped` booms are included and counted.
- `QUANTITY_GROUPS` maps every row of `boom_stats`, `means` and `slow` to exactly one group.
- The wide export round-trips its column naming, and the monthly files concatenate to the
  same frame as a single pivot of the whole test period.

### Phase 9: Post, and finalizing the schema

**`post/classify.py`**: `IntervalClassifier(classes: list[tuple[name, interval]])`. Intervals
use the old `SingleClassifier` syntax (`(a,b)`, `[a,b)`, `inf`, `-inf`); port its parsing from
`windprofiles/utilities/classify.py`, not its class hierarchy. `.classify(values) ->
np.ndarray[object]` gives None for NaN or no match. `stability_classes(pairs, cfg_post) ->
Series[slot → class]` uses `rib` for `post.stability.pair`.

**`post/report.py`** (`ttu-report --stability`):
- `stability_yield.csv`: per class × boom × variant, the fraction of computed slots whose
  ustar, (wvpts, cov) and ti survived filtering.
- `stability_flags.csv`: per class × boom × test, flagged-sample fractions.
- `stability_tau.csv`: per class × boom × variant, the τ source and status distribution and
  the median selected τ.

**Finalize** output-schema.md and config-reference.md against the implemented code; list every
difference in the phase report.

**Acceptance:** interval parsing (open/closed, infinities, malformed → error); classification of
edge values; the reports are produced on the Phase 6 real-data test output.

### Phase 10: Validation and the first production run

**Goal:** the first-run validation of validation.md §3, then the production run.

1. **`validation/synthetic.py`**, grown during Phases 2–7 (generators listed in validation.md
   §2), and **`validation/checks.py` with `ttu-validate`**: one check per item in
   validation.md §3, each writing CSV tables to `<run dir>/reports/validation/`.
2. **Pilot run (approved):** one month (suggest 2014-04), all booms, all stages. Run
   every check and report as in validation.md §4. Recalibrations are config changes that
   Elliott decides.
3. **Production run:** the full year, locally or on HPCC (§6), with the config Elliott approves.
   Then all reports and checks; a final report of yields, timings and any anomalies.

---

## 5. Porting map

Read the source before porting. Port only what is listed; the rest of the old code is
reference.

| New module | Source (read-only) | Port | Notes |
|---|---|---|---|
| `constants.py` | old `definitions.py` | booms, heights, location, tilt tables, header maps, source units, timezone | Drop plotting and legacy tables. |
| `primary/stage_a.py` | old `processing/primary_ops.py`; `windprofiles/processing/units.py` | `fix_sonic_tilt_misalignment`, `compute_winds` (sonic part), unit constants | Propeller code not ported. |
| `math/polar.py` | `windprofiles/lib/polar.py` | `wind_components`, `polar_wind`, `bearing_to_vector`, `vector_to_bearing`, `signed_angular_distance`, `series_signed_angular_distance` | `directional_rms` is replaced by `yamartino_std` (new, Phase 4). |
| `math/fits.py` | `windprofiles/lib/stats.py` | `ls_linear_fit`, `ls_weighted_linear_fit`, `constrained_linear_fit`, `power_fit` (weighted), `log_fit`, `constrained_log_fit`, the NaN-dropping helpers | Keep `test_power_fit_weighting_reduces_bias_vs_nls`. |
| `math/stats.py` | `windprofiles/lib/stats.py` `autocorrelations`, `detrend`; `processing/sonic.py` `integral_time_scale` | as `autocovariance` (FFT with numpy; statsmodels dropped), `linear_detrend`, `efolding_integral` (e-folding only); `moments` is new (Phases 2, 4) | Keep the lag-count fix's intent (never request more lags than samples). The correlation, agreement, Weibull and sine-fit helpers are not ported. |
| `physics/thermo.py` | `windprofiles/lib/atmos.py` | `pressure_to_slp`, `pressure_above_msl`, `saturation_vapor_pressure`, `water_partial_pressure`, `water_air_mixing_ratio`, `specific_humidity`, `virtual_temperature`, `potential_temperature`, `virtual_potential_temperature`, `dewpoint_temperature`, `vpt_from_3` | AERK Magnus instead of Tetens; `isa_pressure` is new (Phase 0). |
| `physics/most.py` | `windprofiles/lib/atmos.py`, `lib/stats.py` | `friction_velocity`, `obukhov_length`, `businger_dyer_phi`, `most_wind_gradient`, `neutral_loglaw_fit`, `constrained_neutral_loglaw_fit` | The Businger–Dyer functions are for analysis; no stage uses them. |
| `physics/richardson.py` | `windprofiles/lib/atmos.py` | `bulk_richardson_number` (with the scalar fix) | The flux Richardson number and its finite-difference gradients are not ported (design-reference B). |
| `physics/turbulence.py` | old `processing/tertiary_ops.py`, `primary_ops.compute_transport_efficiencies` | the four anisotropy functions (Phase 7), the TE formula (Phase 4) | TI, TKE and ILS are new one-line formulas. |
| `physics/earth.py` | `windprofiles/lib/geo.py` | `local_gravity`, `coriolis` | |
| `physics/constants.py` | `windprofiles/lib/atmos.py`, `lib/stats.py` | the universal constants (Phase 0) | |
| `primary/despike.py` | old `qc/tests.py` `_despike_series` | the 0.6745 constant only | The algorithm is new. |
| `primary/qc_windows.py` | old `qc/tests.py` `_rdbin_series`, `higher_moments` | per-window logic | New global window grid. |
| `secondary/derived.py` | old `processing/secondary_ops.py` | which quantities are derived from which inputs | The formulas are in `physics/`; denominator rule. |
| `secondary/sun.py` | old `processing/secondary_ops.py` `mark_night`, `tertiary_ops` sun elevation | as is, at the slot center | |
| `tertiary/mesonet.py` | old `mesonet.py`, `processing/mesonet_merge.py` | whole | Port the tests. |
| `tertiary/filtering.py` | old `filtering/filters.py`, `filtering/report.py` | the per-criterion breakdown | Boom-level NaN, interval queries. |
| `io/rawfiles.py`, `io/convert.py`, `cli/convert_parquet.py` | old `io/raw.py`, `cli/convert_parquet.py` | parsing, conversion, CLI | No `windprofiles`. |
| `io/store.py` | old `io/dataset.py`, `io/guard.py` | fragment write, unified-schema read | |
| `results.py` | old `io/final.py` | the idea of `load_results` only | Keyed by run tag through the registry (§3.10), and loads any table, not just the wide file. |
| `primary/runner.py`, `logs.py` | old `processing/primary_runner.py`, `processing/primary_pipeline.py` (manifest), `windprofiles/user/logs.py` | manifest, hash guard, pool with log listener | Units are (batch, boom), not files. |
| `post/classify.py` | `windprofiles/utilities/classify.py`, old `classifiers.py` | interval parsing, the scheme definitions | |
| tests | old `tests/unit/processing/test_wind_direction_e2e.py`, `test_mesonet_merge.py`; `windprofiles/tests/unit/test_polar.py`, `test_stats.py`, `test_atmos.py`, `test_geo.py` | adapt | |

Paths: old pipeline `C:\Users\ellwalke\Code\old-tower-processing\src\ttu_tower\`; windprofiles
`C:\Users\ellwalke\Code\windprofiles\windprofiles\`.

---

## 6. Performance, memory, HPCC

- **Volume:** about 15,000 files × 10 booms = 150,000 file-boom Stage A/B executions per year;
  525,600 slot-booms of products per variant (×1 to ×2 with `mrd_unexcised`).
- **Expected hot spots:**
  - despiking medians (about 60 windows of 15,000 samples per file-variable, 7 variables);
  - ITS FFTs (8 rungs × 4 variables per slot-boom-variant);
  - `resolution_dropouts` loops (vectorize within windows).

  MRD and the ladder are cheap. Slow-sensor smoothing adds ~0.06 s per file-boom.
- **Expected cost** (from benchmarked kernels; Phase 6 measures it for real):
  - primary ≈ 0.6–1 s per file-boom → ~25–40 CPU-hours per year (allow ×2) → ~1–4 h wall on
    the local 24-core machine;
  - secondary ~5–20 min and tertiary ~5–15 min per year, which is why they run serially.
- **Memory:** one worker holds about 3 Stage A files and 5 Stage B outputs for one boom
  (≈ 50–100 MB). Fragments are written per unit. The wide export is written per month (Phase 8).
- **Measure, don't guess:** the Phase 6 timing report extrapolates to a year. If it exceeds a
  day on the local machine, report it to Elliott; HPCC is the fallback for production. Keep
  the code Linux-clean: `pathlib`, `spawn`, no Windows-only calls, paths only from config.
  `--nproc` sets primary's worker count.
- **Optimization rule:** profile first. Keep the pure-function structure; any optimization
  must keep every invariance test passing bit for bit (§3.7).
