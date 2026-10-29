# `ttu-view`: architecture & implementation plan

## Status (resume here)
**2026-09-28:** phases 1 and 2 are built, checkpoint-tested by Elliott, and revised per his
feedback. See "Phase 1 checkpoint revisions" and "Phase 2 as built" below.

**Phases 3 and 4 are built and have been through two rounds of Elliott's review fixes** (see
"Phase 3 as built", "Phase 4 as built" and the two "Review fixes" sections).

**Phase 5 (linked brushing, find, bookmarks) is built** ("Phase 5 as built").

**Phase 6 is built and awaits Elliott's checkpoint** ("Phase 6 as built"): composite spectra, the
diurnal × month heatmap, an interval anisotropy map, two-run comparison and matplotlib figure
export. PSD waits for the pipeline to compute it. Elliott commits; nothing here is committed by
me.

## Context

Elliott wants a local GUI to inspect `ttu-tower-processing` 2.0 outputs against the 50-Hz data
behind them. The goals:
- year-long zoomable timelines of any quantity (two panels, any booms) with distributions
- click-through from a plotted point to that slot's high-frequency series (left = as measured,
  right = QC applied) or its autocorrelation
- a per-slot inspector with MRD variance/covariance spectra, τ detection, and ITS/ILS at every τ;
  PSD comes later

The design overview was approved on 2026-09-28 and is summarized in Part A. This plan gives the
architecture (Part B) and a five-phase implementation (Part C). Elliott tests after every phase
before the next starts; phases 1–3 matter most.

**Decisions (Elliott, 2026-09-28):**
- Local Qt desktop app; no HPCC.
- Left click shows the series in SI units with tilt correction and rotation to ue/vn/w, and no
  samples removed. The native sonic frame is a toggle option.
- Pipeline changes approved:
  - opt-in `trace` hooks
  - fixing the bounds-flag naming bug found during review (B3.7), with the viewer aliasing
    existing runs
  - writing per-stage `config.toml` + `config.resolved.json`, as output-schema.md §2 already
    specifies
  - fixing the broken `ttu-runs` entry point
- Phases:
  1. main window, timelines, distributions, and the drill-down series view (done; see "Phase 1
     checkpoint revisions")
  2. Slot Inspector
  3. QC/Profile tabs and τ what-if, plus the anisotropy band/map, scatter tab and distribution
     fits (added during phase 3)
  4. QC what-if (added at the phase-1 checkpoint)
  5. linked brushing, find, bookmarks
  6. the rest (run comparison, PSD, composites, and so on)

**Working rules carried over from the implementation workflow:**
- Stop after each phase and wait for Elliott's go-ahead.
- Never commit; Elliott commits.
- Ask before installing packages or running anything long against real data. Read-only opening of
  the real run is fine, but announce it first.
- Terse code comments that match the repo; no doc-file or section references in code or test
  comments.
- Check that new test file basenames are unique across `tests/`.
- Prefer `pd.isna` / `.dropna()` over `is not None` for pandas-derived values.

---

## Part A — Design summary (approved)

### Main window
```
┌─ File  View  Tools  Help ───────────────────────────────────────────────────────────────┐
│ Run [oneyear ▾]  tz [-06:00 ▾]   Go to slot [2013-11-09 12:10 ]⏎   ◀ day ▶  [Y][M][W][D] │
├──────────────────┬───────────────────────────────────────────────────┬──────────────────┤
│ PANEL A controls │  Timeline A (booms colored by height)              │ Distribution A   │
│ quantity (tree+  │                                                     │ hist per boom +  │
│  search), stat,  │───────────────────────────────────────────────────│ stats table      │
│ variant, booms   │  Timeline B (x-linked)                             │ Distribution B   │
│ PANEL B controls │───────────────────────────────────────────────────│ (visible range   │
│ OVERLAYS         │  Overview strip: whole period, range box,          │  or full period) │
│                  │  availability raster (boom × time)                 │                  │
├──────────────────┴───────────────────────────────────────────────────┴──────────────────┤
│ time · slot · boom · value                                    │ worker status / progress │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### Drill-down
*The click behavior below was revised at the phase-1 checkpoint: a left click now opens the slot
viewer in unexcised mode (see "Phase 1 checkpoint revisions").*

- **Clicks on a data point:**
  - **left** = "as measured"
  - **right** = "QC applied" (removed samples ghosted and colored by test)
  - **double-click** = Slot Inspector
- Clicks on empty background keep pyqtgraph's own menu, which includes export.
- A click snaps to the nearest point within about 8 px. If two booms are within 2 px of each
  other, a chooser menu appears; the tool never silently takes the first.
- Which view opens depends on the clicked quantity's **provenance**:

| Clicked quantity | Opens |
|---|---|
| means (ue/vn/w/ws/wd, ts/vpts, t/rh/p) and gusts | that variable's series |
| var/σ/TI/TKE/std | streamwise fluctuations at the selected τ with block boundaries |
| fluxes (cov, u*, L, ζ, te) | both component series plus that pair's cospectrum |
| ITS/ILS/ratios | pooled ACF at the selected rung |
| τ | Spectra tab |
| QC quantities | series plus flag rug |
| Ri_b/profile quantities | Profile tab |

### Slot Inspector
- **Header:** slot, boom and variant pickers with ◀ ▶ stepping, plus a summary strip.
- **Tabs:**
  - **Series:** the 80-min detection window with a flag rug, frames, a τ-block grid, and a
    fluctuation toggle.
  - **Spectra:** 2×4 MRD grid with ±SE bars and the 1–2–1 smoothed curve with its significance
    band. Markers for peak, reversal, τ (heat, momentum, selected), floor and cap. Optional
    overlays: unexcised, all booms, neighbouring slots, ogive. Reserves an `MRD | PSD` switch.
  - **Scales:** ITS/ILS per rung, with τs highlighted and the τ/10 guide; ACFs on demand.
  - **Profile.**
  - **QC.**
  - **Numbers:** includes a Verify button that recomputes from raw and diffs against stored rows.
  - **τ what-if** side panel.

---

## Part B — Architecture

### B1. Package layout
Everything is new under `src/ttu_tower/viewer/`. The Qt-free layer is fully unit-testable
without a display. Qt code lives only in `viewer/ui/`.

```
viewer/
  __init__.py
  run.py          RunHandle: read-only run opening, per-stage meta/config, commit check
  fragments.py    FragmentIndex: slot/boom -> fragment paths; targeted reads
  timeaxis.py     slot <-> unix seconds <-> display tz; slot/sample helpers
  catalog.py      Quantity model + catalog built from the run's own vocabulary
  timeline.py     year-series loading (LRU), overlays data (night, stability, availability)
  slotdata.py     stored per-slot reads (mrd, tau, ladder, flags, numbers)   [phase 2+]
  reprocess.py    Reprocessor: Stage A/B cache, HF windows, ACF, verify
  provenance.py   Quantity -> DrillTarget
  ui/
    app.py          QApplication bootstrap, pyqtgraph config, main()
    style.py        boom colors (viridis by height, like ttu-windprofiles baseplots), test colors
    worker.py       QThreadPool jobs with generation tokens (stale results dropped)
    main_window.py  toolbar, docks, status bar, QSettings persistence
    panel.py        PanelControls (quantity tree+search, stat, variant, booms, y-scale, style)
    timeline_plot.py TimelinePlot (DateAxisItem, per-boom curves, pick, hover, legend toggles)
    overview.py     OverviewStrip (LinearRegionItem + availability ImageItem)
    distribution.py DistributionView (histograms + stats table)
    series_view.py  HF series window (phase 1; becomes the Inspector's Series tab in phase 2)
    inspector/      inspector.py, spectra_tab.py, scales_tab.py, numbers_tab.py,   [phase 2]
                    qc_tab.py, profile_tab.py, whatif.py                            [phase 3]
cli/view.py       `ttu-view [TAG] [--run-dir DIR] [--test]` -> viewer.ui.app.main
```

**Dependencies** (`pyproject.toml`):
- new extra `viewer = ["PySide6>=6.8", "pyqtgraph>=0.13.7"]`
- `pytest-qt` added to `dev`
- script `ttu-view = "ttu_tower.cli.view:main"`

`viewer/ui/app.py` imports PySide6 before pyqtgraph (and sets `PYQTGRAPH_QT_LIB=PySide6`).
This pins the binding despite PyQt5 being present from `wtxmeso`. Matplotlib is not used by
the viewer until phase 5 (publication export, Agg backend).

### B2. Data layer (Qt-free)

**`run.py` — `RunHandle`**
- `RunHandle.open(tag, test=False)` or `RunHandle.from_dir(path)`.
- **Read-only home lookup.** It must not call `io.runs.find_home()`, which creates the home.
  `results.find_run` and `results.list_runs` also call `find_home`, so the viewer can't use
  them either.
  - Add `io/runs.py: locate_home(user_home=None) -> Path | None`, the same search with no
    creation. `find_home` calls it and creates only when it returns `None`.
  - The viewer's own `list_registered(home)` / `resolve_tag(home, tag)` read the link JSONs
    directly.
- **Reads:**
  - each stage's `run_meta.json` (`io.stage.read_run_meta`)
  - `primary/files.parquet`, `primary/slots.parquet`
  - the accepted-file map via `io.rawfiles.accepted_half_hours`
  - the period via `slots.parquet`
- **Config resolution:**
  1. `<stage>/config.resolved.json` → `config.load.config_from_dict`.
  2. Otherwise, the `config` path in the registry link → `load_config`.
     - Catch `ConfigError` and `OSError` as "config unavailable". Test runs use a dummy TOML.
     - Check `primary_hash(cfg, run_meta["package_version"])`, and likewise the secondary and
       tertiary hashes, against each `run_meta.config_hash`. Use the run's version, not the
       installed `__version__`.
     - A mismatch gives a warning banner, and reprocessing is disabled unless the user
       overrides.
     - **Verified read-only on 2026-09-28:** `configs/oneyear.toml` still matches all three
       `oneyear` hashes, so the fallback works for the phase-1 checkpoint.
- **Commit check.** Compare the installed commit with `run_meta.git_commit` and show the
  result in the title bar and inspector header.
  - Extend `io.stage.git_commit` to take `cwd`, defaulting to the package directory rather
    than the process cwd.
  - Add `git_dirty()` (`git status --porcelain -- src`).
  - Record `git_dirty` in future `run_meta`. The oneyear run's secondary and tertiary stages
    report `0f9efe3` even though they were rerun from an uncommitted tree, so the marker is
    needed.

**`fragments.py` — `FragmentIndex`**
- Parses fragment names `bat{h_a:07d}-{h_b:07d}[_b{boom:02d}]` once per table directory into a
  sorted array of `(h_a, h_b)`.
- `batch_for_half_hour(h)` uses bisect.
- `paths(table, booms=None, half_hours=None)` selects files by boom suffix (primary) and by
  batch.
- `read(table, *, booms, half_hours, columns, filter)` builds a `pyarrow.dataset` over just
  those paths, with a unified schema. Unification is required; see `io/store.read_table`'s
  docstring.
  - Refactor `io/store.py`: add `read_paths(paths, columns, filter)`, and have `read_table`
    delegate to it. The behavior of `read_table` is unchanged.
- **Measured on the real run:**
  - boom-8 `mrd`/`ladder` for a whole year, by file selection: 0.06–0.07 s (a whole-directory
    scan took 7.9 s)
  - boom-8 `flags` for a year: 0.9 s
  - tertiary `boom_final` filtered by variable: 0.3–1.1 s
- Flags near a batch edge: `flags_window(boom, g_lo, g_hi)` reads every batch overlapping the
  half-hours of `[g_lo, g_hi)`, because flags are clipped to each unit's core. The pieces
  don't overlap, so `FlagStore` union semantics hold.
- Outage half-hours belong to no batch, so `batch_for_half_hour` returns `None`.
- Non-`bat…` files, such as `slot_boom/outages.parquet`, are indexed separately and always
  included for `slot_boom` (the availability overlay needs them).
- **Legacy flag aliasing.** Existing runs store sonic `bounds` flags under the raw names
  `u`/`v`/`w` instead of `ue`/`vn`/`w` (confirmed in `oneyear`: 5.6M intervals each; see B3.7).
  `flags_window` renames `u`→`ue` and `v`→`vn` for `test == "bounds"` on read.

**`timeaxis.py`**
- `slot_to_unix(k)` = `EPOCH.timestamp() + 600*k` (vectorized); `unix_to_slot(x)` = floor.
- `display_offset_s(tz)` gives the pyqtgraph `DateAxisItem(utcOffset=...)` value. Its sign
  follows `time.timezone`, i.e. positive west; a unit test pins this.
- Reuse `timegrid.slot_to_time` / `time_to_slot` for text entry and formatting.

**`catalog.py` — `Quantity` and `Catalog`**
- `Quantity(frozen)`: `key`, `table`, `variable`, `stat`, `kind` (`boom` | `pair` | `slot`),
  `variants` (tuple), `label`, `unit`, `group`, `circular: bool`, `categorical: bool`.
- Built from the run's own vocabulary.
  - Scan **only the key columns** (`variable`, `stat`, `variant`, and `boom`/`boom2` where
    present; all dictionary-encoded) of **all** fragments of the tertiary tables
    `boom_final`, `pairs`, `profile`, `slot_final`, `tau_final` and `boom_labels_final`.
    Then group and dedupe.
  - Cache the result in memory, and on disk under
    `<home>/viewer/cache/<tag>-<tertiary config_hash>-<run_summary mtime>.json` (never inside
    the run directory). Roughly 1–3 s once.
  - A single fragment is not enough: vocabulary differs by batch (inner merges in
    `selected_stats`, optional mesonet, skipped batches).
  - `stat` is null for slow and derived rows, so test it with `pd.isna`.
- Add static entries for long tables that the wide export lacks:
  - primary `coverage` (variable × layer)
  - `slot_qc`
  - secondary `tau` (per-cospectrum `tau_s`, status, peak/reversal scales)
  - `tau_selected.source_status`
  - per-slot flag fraction per test
  - `mrd_frame.wd_deg`
- A static `_META` dict maps variable names to label, unit and group (units from
  output-schema §3–§5). Unknown names still appear, under "Other", with the raw name. Nothing
  listed can be missing, and nothing present is hidden. `aniso_class` from
  `boom_labels_final` is included as categorical.
- Loads come from long tables, so no wide-column parsing is needed. The pre-filter overlay
  (phase 3) uses the same keys on `boom_stats` / `means` / `slow`.

**`timeline.py`**
- `load(run, quantity, variant, booms) -> dict[boom|pair|None, (x_unix, y)]`. It is sorted
  by slot and NaN where filtered.
- Caches are owned by each `RunHandle` (a small LRU dict), not module-level `lru_cache`, so
  closing a run frees its data.
- Flag fraction per slot: `flags.FlagStore(frame).fraction(test, boom, var, starts, ends)`
  over slot supports, where `starts = 30000*k`.
- Categorical values (τ source/status) map to integer codes plus a legend.
- **Overlays:**
  - `night(run)` from tertiary `slot_final`
  - `stability(run)` via `post.classify.stability_classes(pairs, cfg.post)`
  - `availability(run)`: `slot_boom.status` as a (boom × slot) uint8 image

**`reprocess.py` — `Reprocessor`** (the only place raw data is touched)
- **State:**
  - `cfg`: the run's resolved config
  - `accepted = accepted_half_hours(files)`. Paths are absolute; a missing file raises a clear
    "raw data moved?" error.
  - an LRU of `StageA` per `(boom, h)` of about 64 entries (~5.7 MB each)
  - an LRU of `StageBOut` plus trace per `(boom, h)` of about 24 entries (~22 MB each once the
    trace is core-sliced and deduplicated, see B3.6), so about 0.5 GB. The size is
    configurable.
- **Concurrency.**
  - An in-flight map `(boom, h) → Future` under a lock, so two threads never compute the same
    Stage B.
  - The lock guards only the map and caches, never the computation.
- **Loading:** `load_boom(path, boom, tmpdir)` → `stage_a(raw, boom, cfg.qc)`, mirroring
  `primary/stream.py:64-83` exactly: `BadFileError` → `None`, and `build_span(A, h, margin)`
  (spans are 150,000 samples: core ± 10 min).
- **Half-hours needed, per window:**

  | Call | Stage B for | Stage A for |
  |---|---|---|
  | `qc_applied` over `[g_lo, g_hi)` | each half-hour overlapping the window (≤2 for 30 min, ≤4 for 80 min) | each of those half-hours ±1 |
  | `acf` | `_half_hours_overlapping(lad_g0, 60000)`, i.e. ≤2 | |
  | `verify` | h−2..h+2 | h−3..h+3 |
- **API:**
  - **`as_measured(boom, g_lo, g_hi, frame="earth"|"sonic")`**
    - builds `to_si` → `couple_triplet` → `tilt_correct` → `rotate_to_earth` from the public
      pieces in `primary/stage_a.py`, skipping `apply_bounds`
    - with `frame="sonic"`, stops after `to_si`
    - t/rh/p in SI; `vpts` from `primary/slow.vpts`
    - needs no Stage B and is cheap: 5 ms per file
  - **`qc_applied(boom, g_lo, g_hi, variant)`**
    - **Returns:**
      - `final_series` plus per-variable final masks (trace), and the **family** masks the
        pipeline actually uses for statistics: momentum for ue/vn/w; ts for ts and vpts; heat
        is also available
      - `filled_mask`: samples interpolated by `fill_short`, ≤1 s. A removed spike ends up
        *filled* and finite in `final_series`, so these samples are ghosted as "filled".
      - smoothed t/rh/p with their usable masks
      - the despike band: **`m ± z·MAD/0.6745`** (outlier test at `despike.py:158`), with the
        per-variable `z_threshold[var]`
    - The `unexcised` mode returns `unexcised_series` / `unexcised_masks`.
    - Assembly uses the public `products.assemble_samples` (B3.5).
    - **Flags come straight from `B[hh].flags`** (in memory), not from Parquet. Those are the
      stored rows by construction.
  - **`acf(boom, k, variant)`** (phase 2)
    - B for the ≤2 half-hours overlapping the ladder window →
      `products.assemble_series_window(B, variant, g0, n)`
    - then `rawrung.rung_its_te(..., trace=t)` → pooled and per-block ACFs per rung and
      variable
  - **`verify(boom, k)`** (phase 2): `file_products(h, B, boom, cfg)` filtered to slot k,
    diffed against stored rows by key; reports the max abs/rel difference per table.
- **`HFWindow` dataclass:**
  - `g0`, `n`, `mode`, `boom`, `slot`
  - `series{var}`, `masks{var}`, `family_masks{fam}`, `filled{var}`
  - `removed{test: {var: bool ndarray}}`
    - built with `FlagStore(flags).mask([test], boom, var, g0, n)`, which already folds in the
      boom-level (null-variable) tests for ue/vn/w/ts
    - vpts is queried as `ts`
    - plus a derived `"coupled"` row: samples of vn/w (or ue/w, …) NaN'd by
      `couple_triplet` because another component spiked, which has no flag row of its own
  - `slow_smoothed{var: (x, usable)}`, `despike_band{var: (lo, hi)}`
- **Cost measured on the real run:** A ×7 = 0.08 s; B = 0.65 s per half-hour; products per file
  0.33 s. Phase 1 series view with its default window: ≤2 B ≈ 1.3 s cold, <0.7 s warm per
  neighbouring file.

**`provenance.py`**
- `drill_target(quantity) -> DrillTarget(kind, variables, needs_tau, tab)` is a static table
  (Part A). The kinds are `series`, `fluctuation`, `flux`, `acf`, `spectra`, `profile`, `qc`
  and `None`.
- Phase 1 implements `series` and `qc` fully. Every other kind falls back to `series` over the
  quantity's input variables, then upgrades in phase 2/3.

### B3. Pipeline changes
Each is small and tested. Results must stay byte-identical unless noted.
1. **Config copies.**
   - `io/stage.py: write_config_copies(stage_dir, cfg, config_path)` copies the TOML if it
     exists and writes `config.resolved.json` via `config.load.to_resolved_dict` with
     `write_json_atomic`.
   - Call it right after `write_run_meta` in `primary/runner.py:86`, `secondary/runner.py:96`
     and `tertiary/runner.py:144`.
   - Tests: a unit test in `tests/unit/io/test_stage.py`, plus assertions in the three stage
     acceptance tests that the files exist and that `config_from_dict(json)` round-trips to
     `cfg`.
2. **`ttu-runs`:** add `cli/runs.py`, which parses `--test` and prints
   `results.list_runs(test=...)` (plan.md §2.3 row). Test it with `capsys`.
3. **Read-only home lookup:** add `io/runs.locate_home()` (no creation). `find_home` calls it,
   then creates if `None`. Behavior is unchanged; add a test.
4. **`io/store.read_paths`**, as in B2.
5. **Public window assembly in `primary/products.py`:**
   - rename `_assemble_samples` → `assemble_samples`
   - add `assemble_series_window(B, variant, g0, n)`, which computes its own half-hours via
     `_half_hours_overlapping`
   - It **validates `variant`**:
     - `mrd` / `naive` → the excised series
     - `mrd_unexcised` → unexcised
     - anything else → `ValueError`
     The private helper treats every non-`mrd` value as unexcised
     (`products.py:84-85`), which is a trap.
   - Existing callers keep working.
6. **Trace hooks.** Each adds an optional `trace: dict | None = None`. When it is `None`,
   nothing is recorded and nothing changes.
   - **`stage_b.stage_b(span, h, boom, cfg, trace=None)`.** Everything it records is
     **core-sliced** and **deduplicated**, to keep a cache entry around 22 MB instead of ~60 MB:
     - `spike`, `excursion`, `unchecked` masks per variable
     - `despike_ref[var] = (m, mad)` as float32
     - `filled_mask[var]`
     - `slow_smoothed[var] = (x, usable)`
     - `l1_masks[var]`, `final_masks_var[var]`
     - `coupled[var]`: NaN after coupling but not after its own despike
     - **Not stored:** `input` and `despiked.x`, which are recoverable from `StageA` + masks,
       and `filled` for sonic/ts, which equals `final_series`.
   - **`despike.despike(..., trace=None)`** records the interpolated per-sample `m`/`mad` it
     already computes. This is free.
   - **`rawrung`.** Split `_pooled_its` into `pooled_acf(fluctuations, max_lag) -> rho | None`
     (None when `pooled[0] == 0`) and the existing `(1.0 / SAMPLE_HZ) * efolding_integral(rho)`
     expression, unchanged.
     - `rung_its_te(..., trace=None)` records `acf[rung_s][var]` and
       `block_acfs[rung_s][var]`.
   - **`detect.detect(..., trace=None)`** fills the trace step by step (`scales`, `lo`, `hi`,
     `u`, `i_top`, `d_smooth`, `se_smooth`, `p`, `reversal`, `k_se`).
     - The early returns mean keys may be absent for `no_data`, `weak` or `unresolved`. The UI
       must tolerate that.
   - **Tests for each:**
     - outputs are identical with and without the trace, on existing fixtures
     - `(1.0 / SAMPLE_HZ) * efolding_integral(trace rho) == its` exactly, using the same
       expression
     - the smoothed curve at `p` passes the significance test
     - the per-variable final masks AND-ed per family equal `StageBOut.final_masks`
7. **Bounds-flag naming bug (found in review, confirmed on `oneyear`).**
   - `stage_a.apply_bounds` keys its removal masks by the raw names `u`/`v`/`w`
     (`stage_a.py:38-44`), and `stage_b.py:142-143` writes those names into `flags`.
     `flags.TESTS["bounds"]` and `tertiary/filtering.py` query `ue`/`vn`/`w`.
   - **Tertiary numbers are unaffected:** the sonic bounds mask is shared by u/v/w, and `w` is
     queried under its correct name, so the momentum/heat bounds fraction is the same.
   - Every `ue`/`vn` bounds query returns nothing, though.
   - Fix: map `u`→`ue` and `v`→`vn` when `stage_b` writes bounds flags, with a regression
     test. Future runs are correct; existing runs are handled by the viewer's read-time alias
     (B2).
   - **Approved by Elliott (2026-09-28): fix + viewer alias.** `oneyear` doesn't need a rerun.
8. **Commit provenance.**
   - `io/stage.git_commit(cwd=<package dir>)` gets a working-directory argument.
   - Add a new `git_dirty()`.
   - `write_run_meta` records `git_dirty`, as an additive key.

### B4. UI architecture (Qt)
- **Model/controller.**
  - `ViewerState(QObject)` holds the run, tz, x-range and the two `PanelSpec`s (quantity,
    variant, booms, y-scale, style), plus signals (`runChanged`, `panelChanged(i)`,
    `rangeChanged`).
  - Widgets observe state; they never call each other directly.
  - QSettings persists state and the dock layout.
- **Worker.**
  - `ui/worker.py`: `run_job(fn, *args, on_done, on_error, key)` wraps a `QRunnable` in a
    shared `QThreadPool` (max 2 threads).
  - Each view holds a generation counter, and results carrying an older generation are
    dropped. This is how navigation "cancels" work.
  - Status-bar progress comes from a job registry.
  - NumPy-heavy Stage B runs in a thread. If the UI noticeably stutters from the GIL, move the
    Reprocessor into a single `multiprocessing` worker process holding the cache. This is a
    contained change behind the same `run_job` API; note it, don't pre-build it.
- **`TimelinePlot`.**
  - A `pg.PlotWidget` with `DateAxisItem(utcOffset=21600)` for `Etc/GMT+6` (positive west)
    and one `PlotDataItem` per boom, with `connect="finite"`, `setClipToView(True)` and
    downsampling.
  - **Verify on day one** that pyqtgraph's `"peak"` downsampling handles NaN. It may use
    `max`/`min`, not the nan-aware versions, which could blank NaN-dense curves at
    full-year zoom.
    - If it doesn't handle NaN, add `viewer/decimate.py`: NaN-aware per-pixel min/max
      decimation, recomputed on `sigXRangeChanged` (debounced), with pyqtgraph's own
      downsampling off.
  - Panel B's x-axis is `setXLink` to panel A. Y auto-ranges to the visible data
    (`enableAutoRange(y, True)` + `setAutoVisible(y=True)`).
  - Mouse handling: subclass `pg.ViewBox` and override `mouseClickEvent`. Double-clicks
    arrive there with `ev.double()` **after** a single-click event.
    1. `pick()` maps the click to data coordinates. For each visible curve it runs
       `searchsorted` on x, checks the neighbours, and computes the pixel distance with
       `viewPixelSize()`, vectorised.
    2. On a hit, `ev.accept()`.
       - A left single click is deferred by `QApplication.doubleClickInterval()`. A following
         double click cancels it and opens the Inspector.
       - A right click fires immediately.
       - Emit `pointClicked(slot, boom, button)`.
    3. On a miss, defer to `super()`, so the right-click menu and export still work on the
       background.
  - The legend reproduces the old tool: click toggles a boom, right-click solos it, and Space
    shows all booms.
- **Distributions.**
  - Debounced (150 ms QTimer) on `rangeChanged`.
  - `np.histogram` per visible boom, with shared bins over the finite values in range; bins
    are lin or log.
  - A stats `QTableWidget` shows N, NaN %, mean, median, σ, p5 and p95.
  - Categorical quantities get a bar chart. Circular quantities get a 0–360 histogram, with a
    rose added in phase 5.
- **Series view.**
  - A `QWidget` window of stacked x-linked plots. The x-axis is seconds relative to slot
    start; the header shows absolute times.
  - Mode toggle: as measured / QC / unexcised / overlay. Frame toggle: earth / sonic, with
    streamwise added in phase 2.
  - Flag rug: one row per test from `removed`, plus "coupled" and "filled" rows.
    - Tests listed in `run_meta.unusable_tests` are filled.
    - `record`-kind tests (`excursion`) are hollow.
    - `bounds`/`spike` removals are always filled.
  - Despike band (`m ± z·MAD/0.6745`) is shown in QC mode.
  - The window reuses itself per click. A "pin" button keeps it and opens new clicks in a new
    window.

---

## Part C — Phased implementation (Elliott tests after each phase)

### Phase 1 — Main window, timelines, distributions, series drill-down
1. Pipeline changes B3.1–B3.8. Only the `stage_b` /
   `despike` traces are needed now, but land all traces together so the test pass is done
   once. Then run the full `pytest tests/unit` and `--slow`.
2. `pyproject.toml` extra, dev dependency and script. In `[tool.pytest.ini_options]`, add
   `qt_api = "pyside6"`, because PyQt5 is also in the venv. **Ask Elliott before
   `pip install -e ".[viewer,dev]"`.**
3. **Day-one spike (throwaway):** plot a full-year NaN-dense series with pyqtgraph peak
   downsampling to settle the `decimate.py` question before building `TimelinePlot`.
4. The data layer: `run.py`, `fragments.py`, `timeaxis.py`, `catalog.py`, `timeline.py`,
   `reprocess.py` (`as_measured` and `qc_applied`), and `provenance.py` (series/qc).
5. The UI: `app.py`, `style.py`, `worker.py`, `main_window.py`, `panel.py`,
   `timeline_plot.py`, `overview.py`, `distribution.py`, `series_view.py`, and `cli/view.py`.
   - The main window has the run picker (from the viewer's read-only `list_registered`), tz toggle, go-to-date, Y/M/W/D and
     ◀ ▶ controls.
   - Overlays in this phase: night shading, stability band, and the availability raster.
6. Default series-view window: `[slot_start − 10 min, slot_start + 20 min)`, which needs ≤2
   Stage B, about 1.3 s cold. A "widen to 80 min" button needs ≤4.

**Phase 1 tests** (`tests/unit/viewer/`, names checked for uniqueness):
- **`conftest.py` has two session fixtures.** Both use `pytest.MonkeyPatch.context()` for
  `TTU_TOWER_HOME`, since a session fixture can't use `monkeypatch`.
  - **`primary_run`** (fast, not slow):
    - primary only, via `validation.synthetic.write_raw_dataset` + `run_primary`
    - 1 boom, about 6 half-hours, `batch_max_files=3`, so there are 2 batches and a batch
      edge
    - one missing file, and via `mutate`: a spike, an out-of-bounds sample, and a
      single-component spike (to exercise coupling)
    - backs the reprocess, fragment and flag tests, so the key correctness checks run by
      default
  - **`full_run`** (`slow`): primary + secondary + tertiary, following
    `tests/unit/tertiary/test_run_tertiary_acceptance.py` (8 half-hours, booms 1–2). It backs
    the catalog, timeline and UI tests.
- **`test_viewer_run.py`:**
  - open by tag and by directory
  - opening a run with `TTU_TOWER_HOME` pointed at a nonexistent path does **not** create it
  - `config.resolved.json` preferred
  - fallback plus hash-mismatch detection
- **`test_fragments.py`:** slot→batch mapping; file selection by boom; edge-batch flag reads
  equal `read_table` + filter.
- **`test_catalog.py`:**
  - every `boom_final` (variable, stat) appears
  - every catalog quantity loads without error
  - no listed quantity is empty in the fixture unless its table legitimately has no rows
- **`test_timeline_data.py`:**
  - `load` equals a direct `read_table` + filter
  - x values match `slot_to_time`
  - flag fractions match `FlagStore.fraction`
- **`test_reprocess.py`:**
  - **QC'd series reproduce stored `means` (the key correctness check).**
    - Iterate over the stored `means` rows, since they are period-filtered.
    - ue/vn/w over the **momentum** family mask, ts/vpts over the **ts** mask, and t/rh/p from
      the core of `slow_smoothed` with its usable mask, all mirroring `slotstats.slot_means`.
    - Assert equality to 1e-12.
  - For the injected spike:
    - as-measured shows the spike
    - QC mode marks it `removed["spike"]` and `filled`
    - the coupled components are marked `coupled`
  - The out-of-bounds sample appears as `removed["bounds"]` under `ue` (via the alias, or the
    B3.7 fix).
  - Warm-cache calls don't reload files; count `load_boom` calls via monkeypatch.
  - Concurrent requests for the same `(boom, h)` compute Stage B once.
- **`test_viewer_ui.py`** (`pytest.importorskip("pytestqt")`; `QT_QPA_PLATFORM=offscreen`
  set in the viewer `conftest.py` before any `QApplication` exists):
  - the main window builds on the fixture run
  - selecting a quantity creates one curve per boom
  - `pick()` returns the right slot/boom for a known point, after `resize`, `show` and
    `qtbot.waitExposed`
  - a double click opens only the Inspector stub, not the series view
  - a left/right `pointClicked` opens the series view in the right mode
  - the distribution updates on `setXRange`
  - `DateAxisItem` tick text in `Etc/GMT+6` matches `slot_to_time`
- **Integration** (`@pytest.mark.integration`; skip if the `oneyear` tag isn't registered):
  - open the real run read-only
  - load `ws` mean for all booms in under 3 s
  - reprocess one boom-8 slot and match its stored `means`

**Checkpoint for Elliott:**
1. `ttu-view oneyear`.
2. Two panels, e.g. ws mean booms 1/5/10 and ustar mrd.
3. Zoom to 2013-11-09 and check that the distributions follow the zoom.
4. Left/right-click boom 8 at 12:10; toggle modes; confirm the flags shown are sensible.
5. Report timings.

### Phase 1 checkpoint revisions (Elliott, 2026-09-28) — implemented
- **Clicks.** A plain left click on a point opens the slot viewer in **unexcised** mode; the
  modes are switched inside the viewer. There is no right-click or double-click drill-down any
  more, and a right click gives the plot's own menu.
  - In phase 2 the slot viewer grows into the Slot Inspector: the viewer is its Series tab, and
    the other tabs are added alongside. There's no separate double-click entry point.
- **Slot viewer:**
  - **frames:** earth (ue, vn, w); **streamwise per slot** (each 10-min slot rotated to its own
    stored mean wind, so v is zero-mean per slot: `reprocess.to_streamwise`); sonic instrument
    (u, v, w), which is only meaningful as measured
  - per-series show/hide checkboxes
  - ◀ ▶ step the focus slot
  - "+ earlier" and "+ later" load neighbouring slots **alongside** (contiguous on a clock-time
    axis, up to 36 slots). The focus slot is shaded darker and slot boundaries are dashed.
  - "single slot" collapses back to the focus slot
  - ±35-min context option
- **Main window:**
  - The overview strip is availability plus the drag-select only.
  - Stability is a separate band x-linked to the timelines, so it scales with the viewed
    interval, with a color legend and an on/off toggle. Night stays a shading toggle.
- **Viewed-interval drift fixed.** Timeline and slot-viewer view boxes are `FixedXViewBox`: x
  autorange never runs continuously, since it would feed back through decimation. The
  auto-range button becomes a one-off fit to the whole period, and programmatic log-mode
  switches don't fit at all.
- **Toolbar:** "Open from directory…", "Raw data folder", "Run folder".
- **Catalog:** the "Site" group is renamed "Mesonet"; `night` is dropped (it's a shading
  toggle); sun elevation moves to "Other".

### Phase 2 — Slot Inspector
- **`slotdata.py`:** per-slot reads of `mrd`, `mrd_frame`, `tau`, `tau_selected`, `ladder`,
  `ladder_coverage`, `means`, `coverage`, `slot_qc`, `boom_stats`, `slow`, `filter_log`,
  `pairs`, `profile`, flags windows, through `FragmentIndex` (single fragment files).
- **`ui/inspector/`:**
  - The **inspector window**: header, summary strip, slot/boom/variant pickers and stepping.
    It opens on double-click and from Go-to-slot; prefetch the neighbouring B.
  - The **Series tab**: `series_view` moves in, with the 80-min window, 20-min window marker,
    streamwise frame (per τ-block rotation using `math.polar.streamwise_angle` /
    `rotate_streamwise`), τ-block grid via `timegrid.majority_block`, and a fluctuation toggle.
  - The **Spectra tab**: 8 plots.
    - Log-x `scale_s`; y lin/symlog/log, with defaults of symlog for cospectra and log for
      variances.
    - ±SE error bars.
    - `detect` rerun with `trace` on stored D/SE/N to draw the smoothed curve,
      significance band, detection range, peak, reversal and τ. Heat and momentum come from
      stored `tau`, the selected τ from `tau_selected`.
    - The run-commit mismatch banner explains any disagreement between stored and recomputed
      τ.
    - Toggles: overlay `mrd_unexcised`, all booms (height-colored), neighbouring slots k±1..3,
      and the ogive (cumulative sum, cross-checked against the ladder's within-block
      covariance).
    - Where `slot_boom.unexcised_computed` is false, primary has no `mrd_unexcised` rows, so
      that overlay shows the `mrd` rows labelled "identical to mrd".
  - The **Scales tab**: ITS (u, v, w, vpts) and ILS vs rung from `ladder` (ILS = ITS × |u
    mean| at the same rung, as `secondary/derived.py` defines it), with τs marked, the τ/10
    line, `ladder_coverage` block counts, and a "Show ACFs" button (`Reprocessor.acf`).
  - The **Numbers tab**: searchable, copyable `QTableView` over all stored rows, plus the
    Verify button.
- **`provenance.py`** upgrades to fluctuation / flux / acf / spectra targets that open the
  matching inspector tab.
- **Tests:**
  - `slotdata` equals a filtered `read_table`
  - spectra-tab τ markers equal stored `tau`
  - `Reprocessor.acf` reproduces stored `ladder` ITS
  - Verify reports zero difference on the fixture
  - inspector UI smoke tests

**Checkpoint for Elliott:** browse slots such as the boom-9 strongly-unstable clipped cases,
compare spectra and τ markers, run Verify on a few real slots, and inspect ACFs.

### Phase 2 as built (2026-09-28)
- **Entry.** A click on a timeline point opens the **Slot Inspector** on the tab that explains
  the quantity:

  | Clicked quantity | Opens |
  |---|---|
  | means, QC quantities | Series |
  | σ/TI/TKE | Series, streamwise frame |
  | fluxes, τ | Spectra, with that spectrum's title highlighted |
  | ITS/ILS/ratios | Scales, with the ACFs computed automatically and those variables drawn thicker |

  - The toolbar's go-to time plus a boom picker and **Inspect** opens it directly.
  - Unpinned inspectors are reused; pinned ones are kept.
  - Stored-data tabs work even when reprocessing is disabled.
- **Header:** ◀ ▶ slot (every tab follows; loaded neighbours move with the focus), boom and
  variant pickers, pin.
- **Summary strip:**
  - slot status
  - selected τ with its source and status, plus each cospectrum's status and τ
  - family coverage
  - ws and wd
  - Ri_b → stability class
  - filter-log failures
- **Series tab:** the phase-1 slot viewer as a panel, plus τ blocks (the selected τ or any rung
  ≤ 600 s, tiling each slot from its start as the ladder does) and "fluctuations" (each block's
  usable-sample mean subtracted).
  - Fluctuations are taken in the displayed frame; streamwise is rotated per slot, not per block
    as the ladder does.
- **Spectra tab** (stored `mrd` only; no raw data):
  - all 8 spectra on a log10 scale axis, with ±SE bars
  - variances on a log y axis (toggle); cospectra linear, since pyqtgraph has no symlog
  - detection rerun with its trace on the heat/momentum cospectra: smoothed curve, ±k·SE
    band, searched range, peak ▲, reversal ✕ with its type
  - stored heat, momentum and selected τ lines; min/max τ dotted
  - toggles for the unexcised overlay, all booms, slots ±3, and a value/ogive mode
  - a red note appears wherever the rerun disagrees with the stored τ (code/config drift)
- **Scales tab:**
  - ITS and ILS vs rung (log–log), with the τ lines and the ITS = τ/`min_tau_its_ratio` guide
  - a per-rung table with the its/its_vpts block counts
  - "Show ACFs": pooled ACF per variable at any rung, 1/e line, ITS markers. Per-block ACFs
    aren't drawn.
- **Numbers tab:**
  - every stored row of the slot, per table: filterable, sortable, copyable as TSV
  - **Verify against raw**: recomputes `file_products` for the slot and diffs means, coverage,
    slot_qc, mrd_frame, mrd, ladder and ladder_coverage by key (flags are not compared, since
    their intervals are split and merged differently per unit)
- **Checked on `oneyear`** (boom 8 slot 728857; boom 5 slot 728893):
  - Verify is exact in every table
  - the detection rerun matches the stored τ
  - the ACFs reproduce the stored ITS exactly
  - timings: inspector 1.6 s, ACF 0.8 s, Verify 3.2 s, slot step 1.3 s

**Post-phase-2 fixes (Elliott's feedback, 2026-09-28):**
- The inspector opens via `QTimer.singleShot(0)` after the click and is activated. Otherwise
  Windows could leave it behind the main window.
- Timeline left axes widen to the longest category label, and the band and overview follow, so x
  stays aligned. Unused categories are dropped.
- Point picking uses `decimate.nearest_on_curve`:
  - every sample within 8 px counts, plus the drawn line segments, snapping to the nearer end
  - segments across the 0/360 seam don't count for directions
  - the hit rate along drawn curves is now 100% at every zoom (it was 63–80% at year and 6-h
    zoom)
- Timeline markers are 5 px, and the default style is lines + points.

### Phase 3 — QC and Profile tabs, τ what-if, QC overlays
- **QC tab:** the coverage layer ladder per variable, flag fractions per test over the slot and
  the detection window, `slot_qc` vs thresholds from `cfg.qc`, and `filter_log` reasons per
  variant.
- **Profile tab:** ws / wd / TI / σ_w / fluxes / VPT / τ vs height. Filtered booms are ghosted.
  Draw the power-law and log-law curves from `profile` (`alpha`, `gamma`, `loglaw_*`), plus the
  Ri_b / lapse / veer pairs.
- **τ what-if panel:** a form over `cfg.secondary.detection` and `.selection` fields, prefilled
  from the run's config. On edit, rerun `detect` and `select` on the stored spectra for the
  current slot (all booms), and show new vs stored markers and a table.
- **Timeline overlays:**
  - Filtered values: the pre-filter value from `boom_stats` / `means` / `slow` where tertiary
    is NaN, drawn hollow grey, with a hover tooltip listing the `filter_log` group/criterion.
  - τ-status markers on second-moment quantities.
- **Tests:** what-if with unchanged parameters reproduces stored `tau_selected`; the filtered
  overlay equals the non-NaN secondary rows where tertiary is NaN; tab smoke tests.

### Phase 3 as built (2026-09-28)
- **Inspector tabs**, in order: Series · Spectra · Scales · QC · Profile · Anisotropy ·
  τ what-if · Numbers.
  - Profile, Anisotropy and what-if read the slot across every boom (`slotdata.load_across`),
    only when one of them is opened.
- **QC tab** (`slotqc.py`), stored rows only:
  - **Coverage ladder:** variable × layer. Usable coverage below tertiary's `min_coverage` is red.
  - **Flag fractions:** test × variable, over the slot, the 20-min ladder window or the 80-min
    detection window.
  - **Slot statistics against their limits:** skew/kurt ranges, the shadow sector, and the
    bounce limits. Tripped tests are red if they remove samples, orange if only recorded.
  - **Tertiary's filtering gates, recomputed with their values:** coverage per required family,
    and bounds/spike fraction per input variable over the slot (±5 min where τ = 1200 s).
    - The note turns red if the failures differ from the stored `filter_log`.
    - Checked on `oneyear`: 60 boom-slots, all identical.
- **Profile tab** (`profiledata.py`):
  - ten panels against height: ws, wd, veer, θv, selected τ, TI, σw, u*, w'θv', wd std
  - tertiary-filtered booms drawn hollow grey at their pre-filter value, with filter_log's
    reasons on hover; the inspected boom ringed; click a point to switch boom
  - the stored fits drawn through the data, over the heights they were fitted to: α power law
    and both log laws on ws, γ on TI, γ_wd on wd std. The coefficient is refit; the exponent is
    the stored one, and a mismatch is noted.
  - tables: Ri_b and dθv/dz for every pair (the stability pair first), and every stored `profile`
    row
- **Anisotropy tab** (`anisotropy.py`):
  - the barycentric map (1C left, 2C right, 3C apex) with the class regions tinted and the
    plane-strain line
  - every boom of the slot, height-coloured; filtered ones hollow at their pre-filter
    eigenvalues; the inspected boom ringed
  - the inspected boom's track over ±N slots (click a point to go there)
  - optionally, its whole-period density behind
  - a table of λ1–3, β, φ and class per boom
  - The map classification reproduces `physics.turbulence._classify_anisotropy_state`, and the
    stored `aniso_class`, exactly.
- **τ what-if tab** (`whatif.py`):
  - a form over `[secondary.detection]` and `[secondary.selection]`, prefilled from the run,
    with the loader's cross-key rules checked live
  - reruns `secondary.derived.tau_tables` (+ `materialize_missing_unexcised`) on the slot's
    stored spectra for every boom
  - a table of stored vs what-if heat/momentum/selected τ, plus u* and σw read off the ladder at
    either τ
  - for the inspected boom: both cospectra with the what-if detection trace and stored (solid)
    vs what-if (dashed) τ lines
  - With the run's own parameters it reproduces every stored τ; this is tested on the fixture
    and checked on 6 `oneyear` slots.
- **Timeline overlays** (panel checkboxes):
  - **filtered values:** hollow grey markers at the pre-filter value (from `means`, `slow` or
    `boom_stats`), wherever tertiary NaN'd a `boom_final` value. On by default.
    - Hover gives filter_log's reasons.
    - A click opens the inspector on the QC tab.
    - Reads only the batches holding a NaN'd slot: 0.5–1.5 s for a year of 10 booms.
  - **τ status:** markers where the selected τ was capped (▲), unresolved (◆), fallback (■) or
    none (✕), on τ-dependent quantities.
- **Clicks now also open:**
  - pair and profile quantities → the Profile tab
  - coverage, slot_qc and slot status → the QC tab (flag fractions still open the Series tab
    with its rug)
  - slot-level quantities → the Numbers tab
- **Anisotropy band:** under the stability band. It shows each boom's `aniso_class` per slot
  from `boom_labels_final`, with a variant picker and per-boom toggle buttons. A click on a cell
  opens the inspector's Anisotropy tab. Toggle it from the controls dock.
- **Scatter tab** in the right dock (`ui/scatter.py`, `curvefits.py`), modelled on
  ttu-windprofiles' `baseplots.compare`:
  - x and y default to panels A and B, each with its own boom/pair dropdown, plus a swap
  - visible range or whole period
  - points or a density map (linear/log counts), coloured by one colour, stability or month
  - log axes
  - Overlays:
    - a linear trend
    - a binned median/mean line
    - y = x
    - the curve-fit catalogue of `curve_fits`: linear, constrained linear (intercept given; was "through origin"), power, log, exponential,
      neutral log law, double linear (and constrained double linear), ζ(Ri) and α(Ri). Each fit takes
      x-percentile limits, optional binning (count, mean/median, minimum per bin), and least
      squares in y or in ln y (relative errors).
  - Statistics: N, Pearson r, Spearman ρ, and Cohen's κ on signs.
  - The view frames the 1st–99th percentiles.
  - A click on a point opens the inspector.
- **Distribution fits** (`distfits.py`) in each histogram's "fit" menu:
  - normal, log-normal, Weibull and gamma (positive-only where needed), or von Mises for
    directions; Rayleigh, exponential and beta were added later (see "Changes")
  - maximum likelihood per boom on at most 50k values, run in the worker
  - dashed PDFs over the histograms, plus a table of parameters, KS and AIC
- **Revisions (Elliott, same day):**
  - **Scatter:**
    - no month colouring
    - it uses the viewed interval unless the right dock's shared "whole period" box is checked,
      exactly like the distributions
    - curve fits run only when **Fit** is pressed, in the worker. A fit whose data or settings
      have since changed stays drawn, dashed and marked "earlier", until refitted. Trend and
      binned lines still follow live.
  - **Wind rose tab** (right dock, `windrose.py`, `ui/windrose_view.py`):
    - roses for one chosen boom and for the mesonet (10 m), over the viewed interval or the
      whole period
    - the tower rose uses tertiary's final values, or "unfiltered": final plus the values
      tertiary filtered out, i.e. primary's means
    - 8/16/36 sectors and speed bins 0–2–4–…–15+ m/s
    - hover a wedge for its share
  - The right dock only refreshes its visible tab; the others catch up when shown.
- `scipy` moved into the `viewer` extra; it's already installed.
- **Tests:**
  - `test_slot_qc_whatif.py`: gates vs filter_log, what-if reproduction, overlay vs secondary,
    τ status, profile
  - `test_viewer_fits_aniso.py`: map classes vs the pipeline, fits recover parameters,
    distribution fits, band vs `boom_labels_final`
  - `test_inspector_tabs_ui.py` (slow): every new tab, overlay clicks, the band, scatter and
    fits
- **Found on `oneyear`** (not a viewer issue): at 2014-08-05 05:50, booms 1–7 are `no_data`,
  although their data are present. QC excised all of it: ts usable_l1 is 0 and momentum usable
  is 0.

### Phase 4 — QC what-if (added 2026-09-28)
- **What.** In the slot viewer, a side panel lists the run's `[qc]` parameters, prefilled from
  its resolved config:
  - `bounds`
  - despiking: `z_threshold` and `min_mad` per variable, window, stride, `max_spike_samples`
  - `[qc.windows]`: resolution/dropout limits, skew/kurt ranges
  - `[qc.second_layer]`
  - `min_coverage`, `max_fill_gap_samples`, `unusable_tests`
  - the slow-sensor smoothing widths

  Editing a value and pressing **Apply** reprocesses the loaded slots with the edited QC. The
  panel then shows what changed:
  - the flag rug gains a "what-if" row per test, with added and removed intervals colored
    differently
  - the ghosted removed samples switch to the what-if set
  - a table compares slot coverage (usable fraction per family) and slot means, stored against
    what-if
- **How.** It fits the existing design; nothing in the pipeline changes.
  - `Config` is a frozen dataclass, so the what-if config is
    `dataclasses.replace(cfg, qc=replace(cfg.qc, ...))`.
  - A second `Reprocessor` holds its own Stage A/B caches, keyed by the what-if config's
    `qc`-section hash, so the stored-config caches stay intact.
  - Bounds live in Stage A, so a bounds edit recomputes Stage A as well. Every other edit reuses
    the cached Stage A and reruns Stage B only: about 0.65 s per half-hour, so about 1–2 s for a
    one-slot view.
  - "Reset" returns to the run's config.
- **Optional follow-on.** Chain the what-if Stage B into `file_products`, then `detect` and
  `select`, to show how a QC change moves the slot's spectra and τ. This joins up with the τ
  what-if (phase 3).
- **Tests:**
  - unchanged parameters reproduce the stored flags and means exactly
  - raising `z_threshold` removes a known synthetic spike from `removed["spike"]`
  - what-if caches never leak into the stored-config caches

### Phase 4 as built (2026-09-28)
- **Where:** a **QC what-if…** button on the Series tab opens a side panel (`ui/qc_whatif_panel.py`).
- **The form:** every `[qc]` setting of the run (`qcwhatif.fields` walks the config, so new
  settings appear on their own), grouped by section:
  - coverage and gaps, and `unusable_tests` as checkboxes
  - bounds
  - despiking, including per-variable `z_threshold` and `min_mad`
  - window tests
  - slow smoothing
  - second layer
  Edited fields are highlighted.
- **Apply:**
  - The edited config is first checked with the config loader's own rules (`config_from_dict`);
    rejections are shown and the last what-if stays.
  - Then the loaded slots are reprocessed with a second `Reprocessor(cfg=what-if)`, which has
    Stage A/B caches of its own.
  - It reads the run's Stage A unless the bounds changed, the only thing Stage A depends on.
  - The run's own caches are never written.
- **Shown after Apply:**
  - The series, masks, ghosts and despike band switch to the what-if.
  - The flag rug gains "… what-if" rows for every test·group that differs: magenta = flagged
    only with the what-if, green = flagged only with the run's.
  - A table compares, per loaded slot, usable coverage (momentum, heat, ts, t, rh, p) and slot
    means (ws, wd, w, ts, t, rh, p): the run's stored rows vs the what-if, with changes
    highlighted.
  - **"τ for the focus slot"** (the optional follow-on, built):
    - recomputes the slot's primary products (`file_products` over h−2..h+2) with the what-if
      QC
    - then runs secondary's `tau_tables` with the run's own secondary settings
    - shows heat, momentum and selected τ plus u*/σw (ladder at τ), stored vs what-if
    - takes about 2.7 s
- **Reset:** returns to the run's settings.
- **Checked on `oneyear`** (boom 8, 2013-11-09 12:10):
  - An unchanged what-if reproduces all 39 stored coverage/mean rows exactly, and the stored τ.
  - Apply takes 1.2 s.
  - Raising p's z_threshold to 10 drops nearly all of the continuous p spike flags.
- **Tests:** `test_qc_whatif.py`
  - unchanged settings reproduce the stored flags (every test and variable) and slot rows
    exactly
  - a higher ts `z_threshold` keeps the injected spike
  - what-if caches never touch the run's, and a bounds edit stops Stage A sharing
  - unchanged settings reproduce the stored τ (slow)
  - plus a UI test in `test_viewer_ui.py`

### Review fixes (Elliott, during his phase-3/4 review, 2026-09-28)
- **Wind rose:**
  - "unfiltered" is now the first-layer means (`slot_qc` ws/wd `mean_l1`), taken before the
    direction (shadow) and bounce removals and tertiary filtering, so the shadow sector shows
    what was removed. This matches ttu-windprofiles' windrose notebook.
  - Dashed lines of constant bearing: comma-separated, by default the shadow sector's edges.
  - Both roses share one radial scale.
  - A "mesonet: same slots" option restricts the mesonet rose to the tower rose's slots.
- **Night shading** is on by default.
- **Series tab:** a **flag rug** toggle, off by default (the last series then carries the time
  axis). Applying a QC what-if turns it on, since that's where the what-if's differences are
  drawn.
- **Filtered values on the timelines:** a grey ring around an × in the boom's colour.
- **Warnings:**
  - The console warnings were pyqtgraph computing the extent of all-NaN marker series (no data
    in view): benign. They're removed at the source (decimated curves and inspector plots
    draw nothing when nothing is finite).
  - `QWindowsWindow::setGeometry` was real: with 10 booms selected, the timeline headers and
    the band legends made the window's minimum width 1634 px, wider than a 1920-px screen once
    the docks were added.
    - Legend buttons now read "b10", with the height in the tooltip.
    - Titles and legends wrap.
    - The minimum width is now about 1050 px.
  - `ui/logs.py` sends Python warnings (each once per place) and Qt's messages to
    `<home>/viewer/ttu-view.log`, a rotating 2 MB file, instead of the console. Errors go to
    both.
  - The status bar shows "N warnings logged" as a link to the log.
  - After the fixes, a scripted session over every tab on `oneyear` logged none.
- **Scatter α(Ri) fit** (the port of ttu-windprofiles' `curve_fits.fit_alpha_ri`, now listed as
  "alpha-Ri: α(Ri), shear exponent vs Ri"):
  - its neutral band and Ri_c are editable, in a settings row that appears for fits with settings
    of their own (`curvefits.FIT_SETTINGS`)
  - `mean_high_ri` is reported as in the original
  - an empty neutral, stable or unstable range now gives a plain message instead of scipy's
- **Clicks open the inspector for every quantity:**
  - where several booms coincide at the click, it opens the top-drawn one and names the others in
    the status bar (there's no chooser menu any more)
  - points on a flat view (zero-height y range) are picked too
  - a slow test clicks a real point of every catalog quantity and checks the inspector opens at
    that slot
- **Series defaults:** ue, vn, w and sonic θv are checked, the rest unchecked. The series behind a
  clicked value are checked too, e.g. T for a T mean.
- **τ what-if plots:**
  - ± SE bars on the stored spectra
  - dotted min/max τ, at the what-if's values
  - a key line explaining each marking
  - where a family is `no_data` (no usable samples in the slot, so detection stops before
    smoothing), the smoothed curve, band and peak it would see are drawn in grey with a note
  - an empty spectrum says so

### Review fixes, second round (Elliott, 2026-09-28)
- **Right dock** is titled "Summary". Its tabs are Distributions, Scatter, Profiles, Wind rose
  and QC; each uses the viewed interval unless "whole period" is checked.
- **Distributions:**
  - a stability-class subset (all, or one class of the configured Ri_b pair)
  - one table per panel: the summary statistics, with the fit's parameters, KS and AIC appended
    as columns. It replaces the old fixed-height stats and fit tables, whose empty space grew with
    the dock.
  - The table sits under a splitter handle and scrolls. By default it takes at most 30 %
    (`TABLE_SHARE`) of the height it shares with its histogram, and never more than its rows
    need. Once its handle is dragged, the user's height is kept.
- **Profile error bars** have caps. By stability, each class is drawn a few pixels
  (`OFFSET_PX`) above or below the others at a height, so the bars don't overlap. The offsets
  are recomputed in view units on zoom or resize, with the height range fixed so they can't feed
  back into it; the fitted curves aren't offset.
- **Profiles tab** (`viewer/profiles.py`, `ui/profiles_view.py`):
  - Panels A and B side by side, for per-boom quantities; pairs, slot-level values and
    categories say they have no profile.
  - Each boom's mean (± σ) or median (± MAD), over the interval. Directions use the vector mean
    and Yamartino σ, and median is greyed for them.
  - Points by default; lines, spread bars and log height are toggles.
  - "by stability" draws one profile per class in class colours, with a key above the plots.
  - A fit per panel: power law, log law, linear or logarithmic, fitted against height to each
    profile. Wind speed and TI (`ti`, `ti_u/v/w`) default to power law, others to no fit.
    α (or u*, z0, …), booms used and RMSE per profile go in a table.
- **QC tab** (`viewer/qcsummary.py`, `ui/qc_summary.py`), per boom:
  - **availability:** % of slots computed, no data, no file or missing, plus the momentum and
    heat usable fractions
  - **samples left at each coverage layer**, for a chosen variable
  - **% of samples flagged by each test**, for a chosen variable; boom-level tests count toward
    ue/vn/w/ts
  - **% of computed slots tertiary filtered**, per group, for a chosen variant; hover a cell for
    the split by criterion
  - **how τ was chosen**, by source status
  - Loading: coverage, filter_log and tau_selected take about 4 s. Flags take about 25 s for 10
    booms, counted per slot by one weighted `bincount` per fragment (FlagStore took 9 s a boom).
    Both are read once per run.
- **Boom defaults:**
  - every boom is checked in both panels
  - pair quantities remember their own checked pairs
  - Ri_b and dθv/dz start on 2–4
  - veer starts on every X–4 pair except 4–4 (`panel.default_pairs`)
- **◀◀ ◀ ▶ ▶▶:** ◀◀ ▶▶ step by the whole viewed interval (the old arrows stepped by 80 %),
  ◀ ▶ by half of it.
- **QC tab, τ:** found, capped and unresolved are each split by the cospectrum the τ came from
  ("found · momentum", "found · heat", …); fallback, fixed and none stay whole.
- **Whole numbers print in full** (10123, not 1.01e+04) wherever a count is held as a float, e.g.
  beside NaNs: `tables.number` is used by every table fill, the Numbers and Scales tabs and the
  timeline hover.
- **Legend buttons:** a shown boom has a dark outline around a pale fill instead of a dark fill.
- **Overlay key** reads "⊗ filtered".
- **Greyed controls:**
  - Distributions: log bins for categories, directions or non-positive values
  - Panels: line style for categories
  - Scatter:
    - colour-by in density mode, and its stability option without a stability scheme
    - the binned-line settings unless "binned" is on
    - the fit-bin settings without bins
    - Fit with no curve chosen; choosing "none" clears a fit
  - Wind rose: "mesonet: same slots" without mesonet data
  - Profiles: median for directions, and "by stability" without a scheme
  - Series: the despike band when as measured, fluctuations without τ blocks
  - Spectra: ± SE and detection details in ogive mode, the unexcised overlay when already unexcised
  - Scales: Show ACFs without reprocessing
  - Inspector Profile: fits without a config
- **Clicks that didn't open the inspector:** a click that landed exactly on a drawn marker was
  taken by pyqtgraph's ScatterPlotItem, so the plot never saw it; near-misses worked. The
  timeline's marker items now pass clicks through (`curves.pass_clicks`). A test clicks a marker
  through Qt's real mouse events.
- **Opening feedback:** from the click until the inspector has read its slot, the app shows a
  busy cursor and "opening the Slot Inspector: boom b, time…". Its reads jump the job queue
  (`JobRunner` priorities), and a superseded job that hasn't started yet is dropped from the
  queue instead of run.

### Phase 5 — Brushing, find, bookmarks
- (The scatter tab was built in phase 3.) Linked brushing: a lasso on the scatter highlights
  those slots on the timelines, and vice versa.
- A find bar: a pandas `query` over a frame assembled from catalog quantities, plus a "random
  sample of slots flagged by test X on boom b" preset. The result list drives inspector ◀ ▶.
- Bookmarks as JSON under `<home>/viewer/bookmarks.json`, created on first save, with export.

### Phase 5 as built (2026-09-28)
- **Linked brushing** (`viewer/brush.py`):
  - Shift-drag on a timeline selects the slots of the dragged time span.
  - Shift-drag on the scatter draws a lasso around points (`brush.in_polygon`, in view
    coordinates, so log axes work).
  - The toolbar's **Brush** toggle (B) makes plain drags brush.
  - Ctrl adds to the selection, Alt removes from it, Esc clears it.
  - The selection is one set of slots shared by every view (`MainWindow.set_selection`):
    - magenta rings on each visible curve's value at those slots
    - a "sel" row in the whole-period overview, showing where in the year they fall
    - each boom's histogram split into the selection (filled) and the rest (lines), each a
      density of its own slots on shared bins, so their shapes compare (revised at Elliott's
      request: the selection was first drawn as a share of the whole and came out very short).
      A second table then gives the selection's statistics (captioned "brushed selection
      (filled)") under the rest's ("the rest (lines)"). A distribution fit is made to each part
      separately: its parameters go in that part's table, the rest's PDF is dashed and the
      selection's dash-dotted. Fit columns that share a name with a sample statistic get "(fit)"
      (a normal fit's σ).
    - ringed points on the scatter
    - a "selection vs rest" grouping in Profiles, chosen automatically when a selection first
      appears
  - The status bar shows the count ("N slots (h) in M stretches") with **Step through**, which
    opens the inspector on the selected slots for the go-to (or last inspected) boom, and
    **Clear**.
- **Find** (`viewer/find.py`, `ui/find_panel.py`; the left dock's Find tab):
  - A pandas query (`DataFrame.eval`, python engine) over one row per (slot, boom).
  - Every catalog quantity has a name, e.g. `ws_mean`, `ustar`, `rib_2_4` (one per pair),
    `tau_status`, `heat_tau_s`, `flag_spike_w`, `coverage_momentum_usable`, `qc_w_kurt`,
    `status`.
  - Built-ins: `slot`, `boom`, `height`, `time` (wall clock in the run's zone, so
    `time > '2014-01-05'` works), `hour`, `month`, `doy`, `stability` (class name), and
    `filtered` / `filtered_<group>` (from filter_log, for the chosen variant or the means).
  - Only the names a query uses are read, and a top-level `and boom == N` / `boom in [...]`
    limits the reads to those booms (`find.pinned_booms`).
  - Unknown names get "did you mean …"; a non-condition says so.
  - Options: a variant (for variant-specific quantities and filtering), "viewed interval only",
    and a random sample of N.
  - Presets fill in templates, among them "slots a test flagged, 30 at random". **Names…**
    lists every column, searchable, and inserts one on double-click.
  - Results are listed (time, boom and the named columns). **Step through**, or a
    double-click, opens the inspector at that match; **Select on plots** makes the matches the
    brushed selection; **Save as list…** keeps them as bookmarks.
- **Inspector sequences:** "◀ match / match ▶" with "3 / 30 label" step through a list of
  (slot, boom): find results, a bookmark list or the brushed selection. Outside the list they go
  to the nearest match in time. An ordinary click-through clears the sequence.
- **Bookmarks** (`viewer/bookmarks.py`, `ui/bookmarks_panel.py`; the left dock's Bookmarks
  tab):
  - The inspector's **☆ Bookmark** saves the run, slot, boom and variant with a note, to a
    named list (pick one or type a new name).
  - They're kept in `<home>/viewer/bookmarks.json`, which is created on the first save and
    written through on every change.
  - The tab lists this run's bookmarks (or every run's) by list. Notes and list names are
    edited in place.
  - Buttons: **Open** (steps through the listed bookmarks, each in its own variant), **Select
    on plots**, **Delete** (confirmed) and **Export…** (CSV with a time column, or JSON).
- **Flag counting** moved to `viewer/flagcounts.py` (the per-fragment weighted `bincount`), now
  also behind the timelines' flag-fraction quantities and find's `flag_*` columns. They were
  ~9 s a boom with FlagStore and are now 1–3 s a boom, still equal to FlagStore's counts.
- **Tests:**
  - `test_viewer_find.py`: lasso geometry, spans and combining, the bookmark store (persist,
    edit, remove, export), query names, boom pinning, and find against the stored values,
    built-ins and filter_log (slow)
  - slow UI tests: a real Shift-drag on a timeline and a lasso on the scatter, find stepping
    the inspector, and bookmarks saved, edited and opened (on a temporary store, never the real
    file)

### Phase 6 — Later
- Composite spectra over the visible range or a stability class.
- Two-run comparison.
- Diurnal × month heatmap and wind rose.
- Barycentric anisotropy map.
- Publication export (matplotlib).
- PSD, once the pipeline computes it, in the reserved Spectra switch.
- (Distribution fits were built in phase 3.)

---

### Phase 6 as built (2026-09-28)
The right dock's tabs are now Distributions, Scatter, Profiles, **Spectra**, **Diurnal**,
**Anisotropy**, Wind rose, QC and **Compare** (scroll buttons when narrow). Each follows the
viewed interval or the whole period like the others.

- **Spectra** (`viewer/composite.py`, `ui/composite_view.py`): composite MRD spectra like
  ttu-windprofiles' `baseplots.cospectra`.
  - Per boom (toggle buttons), the median over slots at each averaging scale, with 25–75 %
    bars, on a log τ axis. Dashed guides at 60, 180, 600 and 1800 s.
  - Choices: any of the 8 spectra; mrd or mrd_unexcised; all slots, one stability class, or the
    brushed selection (or the rest).
  - "normalize" divides each slot's spectrum by its sum over scales; log y is for variances
    only.
  - A boom's slot × scale matrix is read once per (boom, variant, spectrum) and cached, about
    1 s a boom on `oneyear`.
- **Diurnal** (`viewer/diurnal.py`, `ui/diurnal_view.py`): a panel's quantity for one boom (or
  pair) as an hour-of-day × month heatmap in the display time zone, pooling years.
  - Statistics: median, mean, std or count. Directions use the vector mean and Yamartino σ with
    a cyclic colour map; categories show the share in a chosen category.
  - Hover shows the value and N. **Clicking a cell selects its slots** (linked brushing).
- **Anisotropy** (`ui/anisotropy_view.py`, `anisotropy.boom_map`, `anisotropy.hexbin`): one boom's states
  over the interval on the barycentric map, as the old plotting code's `barycentric.py` drew
  them (revised at Elliott's request).
  - **Heatmap (default):** a hexbin of slots per hexagon (gridsize 40), viridis, every hexagon
    filled and clipped to the triangle, with a colour bar per panel; "log colour" is an option.
    `anisotropy.hexbin` reproduces matplotlib's binning exactly (tested), so the viewer and the
    exported figure agree.
  - **Points** instead, with the class regions shaded underneath if wanted.
  - **Split:** all together, **one panel per stability class**, or "brushed selection | the
    rest", which needs a brushed selection. That option's tooltip says how to make one, and it
    is greyed out without one.
  - Split panels use short corner labels, tight view ranges, no layout margins and narrow
    unlabelled colour bars, so the triangles fill their panels; the colour's meaning is in the
    note under the plots.
  - A table gives each panel's share per anisotropy class (the pipeline's regions).
  - A lasso on a panel selects that panel's slots inside it.
  - The export draws the same panels with `Axes.hexbin`, clipped to the triangle, with full
    corner labels on the first panel only (as the reference does).
- **Two-run comparison** (`viewer/compare.py`, `ui/compare_view.py`): the toolbar's
  **compare** picker opens another registered run read-only.
  - Its version of each panel's quantity (same key and variant, if it has them) is drawn dashed
    in the same colours under this run's curves.
  - The Compare tab lists per boom: N in each run, N both, the means, the bias (other − this),
    RMSE, % identical and r (directions wrap to ±180°). A scatter shows other against this for
    one boom, with y = x.
  - On `oneyear` against `pilot_2014_04`: ws is identical; u* differs slightly where τ changed.
- **Figure export** (`viewer/figures.py`, `ui/export_dialog.py`, toolbar **Export figure…**):
  - Exports the timelines (both panels over the viewed interval, night shading, the compared
    run dashed, lines broken at missing slots) or the showing summary tab. Distributions,
    scatter, profiles, spectra, diurnal, anisotropy, wind roses and compare have figures; QC is
    tables and has none.
  - Drawn with matplotlib's object API on the Agg canvas (never pyplot), in the paper's style:
    tab colours, stability colours and markers from ttu-windprofiles' `style.py`, viridis by
    height.
  - Size in inches and dpi are remembered; PNG, PDF or SVG by extension.
  - Each view hands over a plain description (`figure_spec()`), so `render(kind, spec, size)`
    is Qt-free and tested for every kind.
  - `matplotlib` is added to the `viewer` extra. It was already installed, as a dependency of
    wtxmeso.
- **The right-click Export dialog** (`ui/plot_export.py`, installed by `configure_pyqtgraph`;
  Elliott's review):
  - pyqtgraph's own dialog, subclassed. Its "Export options" panel is given room and a stretch
    so it grows with the window.
  - A "Title and axis labels (for this export only)" box, prefilled from the plot (a timeline's
    title comes from its panel). The text is applied to the plot only while the file is written,
    then restored.
  - A "Saved to <path>" message box on completion, or a warning with the error.
  - pyqtgraph's CSV exporter is replaced by "CSV of plot data (labelled columns)":
    - Columns are named by what they are ("b5 (16.8 m)", "b5 (16.8 m): filtered (value before
      filtering)", "…: brushed selection", "…: <other run>", "… normal PDF"), else the item's
      legend name, else the y-axis label.
    - A date axis becomes a "time (<zone>)" index in the display zone.
    - Timelines export every stored sample, not the decimated points drawn; profiles export the
      true heights, not the drawn pixel offsets.
    - Options: separator, significant digits, one row per x (curves side by side, the default)
      or x and y columns per curve (the default for the scatter, profile and compare plots),
      and the viewed x range or all the data.
- **Job runner fix:** the full suite had begun hanging now and then, in about half the runs.
  - Every stack dump showed the GUI thread stopped at an ordinary Qt call (`self.scene()`,
    `QEventLoop.quit()`, a menu's `translate`, a `clear()`) and a worker stopped at an ordinary
    numpy or pandas allocation, neither moving for minutes. GIL contention alone was measured
    and ruled out.
  - Cause: Python's cycle collector runs on whichever thread happens to allocate. A worker
    thread collecting cycles freed closed windows' and cleared plots' Qt objects off the GUI
    thread, which deadlocked PySide. The first dump caught it directly, in a weakref callback
    during `clear()`.
  - `JobRunner` turns automatic collection off and collects on a GUI-thread timer every
    0.5 s instead (pyqtgraph's `GarbageCollector`, which exists for this). The viewer tests'
    conftest turns automatic collection back on when they finish.
  - Verified 2026-09-29: `test_inspector_tabs_ui.py`, which had stalled about one run in two,
    passed five runs in a row, then the full viewer suite passed (176 passed, 2 integration
    skips).
  - Also, `JobRunner` now runs jobs on plain Python threads with a priority queue
    (`worker._Pool`), because a `QThreadPool` deletes its QRunnables on its own threads.
    Results come back through the same queued signal. A superseded job still waiting skips
    its work (a lock-guarded set), priorities are unchanged, and the main window ends the
    workers on close.
- **Settings:** the main window reads `QSettings()` under the application's organization and
  name, the same registry key as before. The viewer tests point that at a scratch folder, so
  they no longer overwrite the user's window geometry and last run.
- **Variant picker:** it starts at `mrd (selected τ)` and keeps the last variant chosen among
  several as quantities change. The τ-independent variant is labelled "slot (no τ)" instead
  of "—", and every variant has a tooltip. For ws, w and ts means, "slot (no τ)" (primary
  `means`) equals `naive` to rounding on `oneyear`. vpts differs because `boom_stats` converts
  it to the measured pressure while `means` keeps the reference pressure.
- **Already built earlier:** the wind rose (phase 3), the inspector's barycentric map (phase 3)
  and distribution fits (phase 3).
- **Not built: PSD.** The pipeline doesn't compute PSDs yet; the Spectra tab's `MRD | PSD`
  switch stays reserved.
- **Tests:**
  - `test_viewer_composites.py`: composite medians and normalization, diurnal cells (time zone,
    categories, directions, cell slots), comparison statistics and direction wrapping, every
    figure kind rendered to PNG, PDF and SVG, and (slow) the spectra matrix and anisotropy map
    against the stored rows
  - slow UI tests: composite groups and the selection, a diurnal click selecting its slots, the
    anisotropy lasso, a run compared with itself (identical everywhere), and figures exported
    from every tab

### Changes (Elliott, 2026-10-02)

- **Panel B's default quantity** is TI (`boom_final|ti|`, `mrd`) instead of u*.
- **The Slot Inspector's Profile tab** adds TKE and ζ = z/L, for 12 panels in 6 columns.
- **Profile tab, 2026-10-09:** adds w'u' (`uw`, `cov`) after u* and ILS u (`ils_u`) last. Both
  follow the inspector's variant, MRD by default, like the other τ-dependent panels. That
  makes 14 panels in 7 columns.
- **Window width, 2026-10-09:** maximized on a 1920-px screen, the window ran off the right edge
  once Jets → Aggregate had been shown. Its one-row control strip lifted the window's minimum
  width to 1990 px, and Qt never makes a window narrower than its minimum.
  - That strip is now two rows, and its summary line wraps.
  - Each dock's contents now sit in a scroll area whose minimum is capped at 320 × 200, so no
    panel can push the window past the screen again: a squeezed dock scrolls instead.
  - The default dock widths are unchanged.
- **Jets → Aggregate bins:** a "bins" box. "Auto" keeps the fixed widths (1 m/s, 0.5 m/s, 0.5 h,
  10 m; half that slot by slot); a number splits each histogram's range into that many equal
  bins. It applies to speed, falloff, duration and estimated-nose height, not to the bar charts.
- **Transport efficiency, 2026-10-12:** primary's `ladder` table holds TE (`uw|te`, `wvpts|te`) at
  all eight rungs (9.375–1200 s), per variant. A rung's TE uses fluctuations about that rung's
  block means, so like the ogive it is cumulative up to that scale.
  - **Slot Inspector → Spectra:** a fifth column holds u'w' TE (top) and w'θv' TE (bottom)
    against rung, with the same overlays (all booms, slots ±3, unexcised) and τ lines. A
    "transport efficiencies" box hides it (see the plot-group boxes below). The bundle's `te` rows cover this slot
    for every boom, and this boom's ±3 slots.
  - **Summary → Spectra:** "u'w' transport efficiency" and "w'θv' transport efficiency" in the
    spectrum list show the median per boom and rung, with IQR bars, over all slots, a stability
    class or the selection. "Normalize" and "log y" are greyed out for them.
- **Wrapping panel tabs, 2026-10-12:** the Controls and Summary tabs (`ui/wrap_tabs.py`,
  `WrapTabWidget`) wrap onto more rows when the panel is narrow, instead of being cut off. Each is
  a QTabWidget with its own tab bar hidden, under a flow layout of tab buttons.
  - The scroll area that stops a panel widening the window past the screen now wraps only the
    pages (`FitScroll`), not the whole dock. The buttons therefore always span the visible
    width, and a page narrower than it needs scrolls beneath them.
- **Axis units, 2026-10-12:** no plot axis uses an SI prefix any more (there was no "m°",
  "mm²/s²" or "mK" to want).
  - `ui/axis_units.py`, installed by `configure_pyqtgraph`, keeps each label's computed unit.
  - Where the largest value on screen is below 0.01 or at least 10⁴, the ticks show the mantissa
    and the label a power of ten, e.g. "(×10⁻³ K m/s)".
  - Date axes, log-decade axes, fixed-tick axes and unlabelled axes are never scaled.
  - Summary → Spectra's y-axis now carries the spectrum's unit.
- **Flag rug labels, 2026-10-14:** rug rows with longer names lost their labels, e.g.
  `dropouts · sonic`, `excursion · sonic` and `coupled · sonic`. pyqtgraph drops any tick label
  wider than its axis, and the Series panel's left axes are fixed at 72 px. The panel now widens
  every plot's left axis to the longest rug name, so the labels show and the time axes stay
  aligned.
- **Right button, 2026-10-15:** a right drag still rescales and a right click still opens the
  plot menu, but the menu no longer pops up after a hesitant rescale.
  - pyqtgraph called any release before 5 px of movement a click, however long the hold.
  - `ui/right_button.py`, installed by `configure_pyqtgraph`, starts a right drag after 3 px.
  - It opens the menu only on a quick right click: a release within 0.35 s of the press. A
    longer hold that never became a drag does nothing. Left-button behaviour is unchanged.
  - **2026-10-15, second pass:** the menu still appeared sometimes, as a drag ended or while the
    button was still held.
    - pyqtgraph decides from its own drag bookkeeping, which skips the mouse moves it throttles,
      so a fast flick or end-of-drag jitter could finish as a "click".
    - Windows also sends Qt a context-menu event after every right release, drags included.
    - A right release is now a click only if the pointer never travelled 3 px from the press,
      over every move including throttled ones, and the button came up within 0.35 s.
    - The plot menu also never opens while the right button is down.
- **Availability strip, 2026-10-16:** it zooms and pans in time on its own, not with the
  timelines: the wheel or a right-drag zooms, a left-drag on the background pans. The boom rows
  stay fixed and the view never leaves the period. A double-click or the A button shows the
  whole period again, and dragging the blue box still sets the timelines' range. The splitter can
  now make it up to 600 px tall (it was capped at 200); its default height is unchanged.
- **Spectra tab plot groups, 2026-10-16:** a "plots:" row of four boxes, all on by default:
  heat & momentum (w'θv', u'w'), other covariances (v'w', u'v'), variances (u'u', v'v', w'w',
  Ts'Ts') and transport efficiencies (u'w' and w'θv' TE).
  - **Layout:** the grid is rebuilt from the checked groups (`spectra_tab.plot_rows`).
    Covariances go above variances with the TEs beside them, as before. Otherwise each group gets
    a row, or everything shares one row when there are four plots or fewer.
  - **Widths:** rows of different lengths share a grid of equal columns, each plot spanning a
    share, so the plots in a row are equally wide and fill it.
  - **"log y":** greyed out while no variances are shown.
- **Normalize, 2026-10-16:** the Spectra tab has a "normalize:" box for each (co)spectrum plot, and
  the τ what-if one each for its heat and momentum plots.
  - A checked plot is divided by its signed sum over modes (`slotdata.normalized`), as the Summary
    composites are. The same factor applies to its ± SE, overlays and detection curves.
  - Because the sum is signed, a negative-flux cospectrum such as u'w' comes out positive.
- **τ what-if smoothing, 2026-10-16:** a "smoothing" box offers the run's 1-2-1, none, 1-1-1,
  1-2-3-2-1 and 1-4-6-4-1, or typed weights (an odd count, centred).
  - `secondary.detect.detect` (and `derived.tau_tables`) take an optional `kernel`, whose default
    `SMOOTHING = (1, 2, 1)` sums in the original order. Pipeline results and config hashes are
    unchanged.
  - The rerun of every boom and the drawn detection use the chosen weights.
- **Boom presets, 2026-10-16:** the panels' boom presets are all · none · odd · even · 1/5/10, as
  compact tool buttons. Push buttons held the Controls panel about 140 px wider.
- **Panel minimum width, 2026-10-16:** in the Controls and Summary tab widgets, only the shown page
  sets the minimum width (`WrapTabWidget._size_to_current`). A wide hidden page, e.g. Find, no
  longer stops the panel narrowing while a narrower one is shown.
- **Bookmark indicator, 2026-10-16:** the Slot Inspector's button reads "★ Bookmarked" (gold) while
  its slot and boom are in any bookmark list. Its tooltip names the lists and notes, and it updates
  on stepping and when bookmarks are added, edited or deleted.
- **Wind rose defaults, 2026-10-16:** boom 4 (10.1 m, the mesonet's height), unfiltered, with
  "mesonet: same slots" on.
- **Summary τ tab, 2026-10-16** (`viewer/taustats.py`, `ui/tau_view.py`): every (slot, boom, variant)'s
  heat and momentum detections and selected τ, read once (about 2 s for oneyear) and cached.
  - **Shared controls:** the variant (mrd, mrd_unexcised); which slots (all, a stability class, the
    brushed selection or the rest); the visible range or the whole period; and the boom bar.
  - **Categories:** τ takes only the 8 rung values, so each detection falls in a category: a rung
    (found or capped), then unresolved, weak or no_data. A selection is a rung or none.
  - **Distributions:** heat, momentum or selected τ as % of slots per category. There is a line per
    boom, or a line per stability class for one boom, with a legend. A table per line gives N, %
    by status (or by source, for the selected τ), and the median, p25 and p75 of τ on the rungs.
  - **Heat vs momentum:** a joint matrix, heat category by momentum category, as % of slots. The
    agreement diagonal is blue. Clicking a cell brushes its slots. A per-boom table gives, where
    both have a τ: % equal, % heat longer, % momentum longer, and the median difference in rungs.
  - **Selection what-if:** the same settings as the Slot Inspector's τ what-if, applied to every slot.
    - **Selection** (rule, priority, fallback τ) applies at once. It reruns secondary's own `select`
      once per distinct pair of detections: a few hundred cases, about 0.1 s for oneyear. With the
      run's rules it reproduces all 826,261 stored selections exactly.
    - **Detection** (peak significance, min scale, min/max τ, smoothing) applies on "Recompute
      detection" (`taustats.redetect`). It reruns secondary's own `detect` on every slot's stored
      heat and momentum cospectra, one background job per boom: about 9 s a boom, 88 s for
      oneyear's 10.
    - Results are cached per (settings, boom), so returning to settings already computed is
      instant. Rows the run marked no_data stay so. With the run's settings it reproduces the
      stored detections exactly.
    - **Comparison:** heat, momentum or selected τ, run (solid) against what-if (dashed). A
      per-boom table gives % of heat, momentum and selected τ changed, the medians, and %
      heat / momentum / fallback, run vs what-if. A button brushes the slots whose compared τ
      changes.
  - The priority boxes, here and in the inspector what-if, keep their choice as a string
    ("heat,momentum"): Qt turns tuple item data into lists, so `findData` with a tuple found
    nothing.
- **Summary Scales tab, 2026-10-16** (`ui/scales_view.py`, `composite.scales_matrices`): the integral
  scales at every ladder rung, composited over slots like the Spectra tab.
  - **Quantities:** ITS of u, v, w or θv, from primary's `ladder` table; ILS = ITS × |u mean at that
    rung|, as secondary and the inspector's Scales tab define it.
  - **Plot:** a line per boom against rung, as the median with interquartile bars or the mean with
    ± SD bars, log y by default. On ITS, a dashed guide ITS = rung / min_tau_its_ratio.
  - **Table:** the plotted value per boom and rung, with N.
  - **Shared controls:** the variant, which slots (all, a stability class, selection, rest), the
    visible range or the whole period, and the boom bar.
  - **Loading:** one read per boom covers all eight quantities (about 15 s for oneyear's 10 booms,
    then cached).
  - **Export:** the figure uses the composite layout, which now takes an optional `xlabel`.
- **Log axes show values, 2026-10-26:** Scatter's log x/y and Distributions' log bins plotted log10
  values on a plain axis, labelled "log10 …". Both now use `axes.SwitchLogAxis`: decade ticks
  labelled with the values themselves, and "(log scale)" added to the axis label.
- **Scatter log bins, 2026-10-26:** a "log bins" box on the fit's binning
  (`curvefits.bin_average(log=True)`): bins equal in log x with geometric centres, positive x only.
  The "binned" central line uses log bins whenever the x axis is log.
- **Either side of zero, 2026-10-26:**
  - **Scatter:** each log axis has a side box: > 0, or < 0 shown as −x (the magnitudes). The
    points, lines, statistics and fits all use those values, and the axis label gains a "−".
  - **Distributions:** a values box: all values, x > 0 only, x < 0 only as −x, or |x|. A side or
    |x| allows log bins and positive-only fits, and the title and axis label say which values.
    It is off for directions and categories.
  - **Log-bin auto range:** it now spans the 0.1st-99.9th percentiles, as linear bins do. It used
    to run from the minimum, and a side of zero always has tiny magnitudes to stretch it.
  - **HTML labels:** labels and titles are HTML, so "<" is escaped there.
- **Power-law weighting (open, 2026-10-26):** the Scatter power-law fit uses `math.fits.power_fit`
  (log space, weighted by y²), so the largest y values steer it.
  - On oneyear b5, σw against ws: b = 0.52, with the curve about 2× above the binned medians at
    low and mid wind. Unweighted log: b = 1.12. Least squares in y: b = 1.00. 20 log bins, then
    unweighted: b = 0.83.
  - Options were set out for Elliott; on hold at his request. Don't change `power_fit` itself:
    the pipeline's α uses it.
- **Saved views, 2026-10-26** (`viewer/views.py`, `ui/view_state.py`).
  - **Contents:** a view holds both panels' choices (quantity, variant, booms or pairs, log y,
    style, the filtered/τ toggles, the legend entries shown), the time window and time zone, the
    overlay toggles, "whole period", the open Controls and Summary tabs, the dock layout, the
    stability criterion, the brushed selection (as slot runs), and every Summary tab's settings.
  - **Summary tab settings:** each tab lists its controls in `STATE_FIELDS`, and `view_state`
    reads and sets them by widget type. `state_skip()` leaves out values that are only derived,
    e.g. Distributions' from/to boxes while auto range is on.
  - **Named views:** "Views" in the toolbar can save this view (name and optional note), open
    any saved view, or delete one. They are kept in `<home>/viewer/views.json` with the run they
    came from, and open on any run.
  - **On another run:** what the run lacks (a quantity, its booms, a tab, a time zone, a stability
    pair) is skipped and listed in the status bar.
  - **Restore timing:**
    - Panels wait for the run's quantity catalogue: a view opened at start-up is applied
      instead of the default panels.
    - Legend entries wait for each panel's data.
    - Summary settings wait for the stability classes to be settled.
    - A tab whose lists fill only once it's shown (Diurnal, Compare) gets the rest of its
      settings when it's next shown.
  - **Where you left off:** `MainWindow(resume="ask")`, which is how `ttu-view` starts, keeps
    each run's state in the viewer settings (`session/<tag>`). It saves on closing or switching
    run, and on opening a run asks whether to resume; "No" opens the default panels. Windows
    made with the default `resume="never"` (the tests) neither ask nor save.
    - Fixed 2026-10-26: the question is asked before the run's quantities start loading. It was
      asked after, so the dialog's event loop delivered them while it waited: the default panels
      went in, and "Yes" left a view no one applied. A "Yes" after they've come applies at once.
    - Also 2026-10-26: at start-up the main window isn't shown yet, so the question
      (`_ask_resume`) is a window of its own, titled with the run, kept on top and given the focus,
      "Yes" the default. Once the viewer is showing (switching runs) it sits on the main window.
- **Scatter: power law with a fixed exponent, 2026-10-26.** A "power law, fixed exponent" fit
  (`curvefits._power_fixed`) takes its exponent b in the fit's settings row (default 0.5).
  - It fits only a, by least squares in y: a = Σ y·x^b / Σ x^2b, over the positive x. With b = 1 it
    is the fit through the origin.
  - On oneyear b5, σw against ws: b = 0.5 gives a = 0.184 (RMSE 0.183); b = 1 gives a = 0.069
    (RMSE 0.159). The free power law, y²-weighted in log space, has RMSE 0.205 (see the open
    weighting question).
- **Scatter: a third quantity z, and a 3D view, 2026-10-26.**
  - **z's source:** panel A, panel B, any of the run's quantities (`set_catalog`, loaded through
    `z_loader` with every member, a searchable list), or the boom height. By height, every boom
    both panels have is stacked, one point per slot and boom; a slot quantity's value sits at every
    boom, a pair quantity can't. The member pickers are then off, and a clicked point opens its
    own boom.
  - **Coloring by z:** "z" in the color choice colors the points over z's 2nd-98th percentiles, or
    symmetrically about 0 ("diverging about 0", linear z only). Points without z are grey. A color
    bar beside the plot uses the same axis as x and y (log decades, ×10ⁿ factor). Figure export
    draws it with matplotlib's colorbar.
  - **3D view** (`ui/scatter3d.py`, pyqtgraph's OpenGL items; PyOpenGL is now in the viewer
    extras): a window of its own that follows Scatter (points shown, log axes and sides,
    selection), even with the tab hidden.
    - Each axis spans its 1st-99th percentiles, scaled to a unit cube. Ticks, titles and grids on
      the three back walls of the starting view.
    - Colors by z, stability class or one color; point size; face-on views of each plane.
    - Text drawn on the GL widget is missed by `QWidget.grab()`; screenshots use
      `grabFramebuffer()`.
  - **Saved views:** z's source, quantity, variant and member are kept. z's member combo is a
    `view_state.LateCombo`, which holds a choice until the loaded quantity's members arrive.
- **Scatter: x percentile lines, 2026-10-26.** "show" beside the fit's x percentiles marks them with
  dashed vertical lines (percentiles of the x shown, as the fit uses).
- **Scatter: one side of 0 on any axis, 2026-10-26.** Each of x, y and z has "all", "> 0" or "< 0".
  - On a linear axis the chosen side keeps its values as they are, and the axis label says so.
  - A log axis shows one side, the negative one as −x. Turning log on with "all" chosen picks the
    positive side, or the negative one where there are no positive values.
  - The statistics, lines and fits use the same points; for z, the other side is "no z".
- **Scatter: double power law fit, 2026-10-26.** A smoothly broken power law,
  a·(x/x_b)^b1·(1 + (x/x_b)^(1/Δ))^((b2−b1)·Δ), fitted in log space (positive x and y only, each
  decade alike): slopes b1 → b2 about the break x_b, over a width Δ in ln x, all four fitted
  (Δ bounded to 1/200 to 1 of the ln x range).
  - On oneyear b5, σw against ws: raw points give 0.36 → 1.46 about 3.6 m/s; 30 log-binned
    medians give 0.69 → 1.07 about 3.9 m/s. Both choose a sharp break (Δ near its lower bound).
- **3D wind rose, 2026-10-26** (`ui/windrose3d.py`): "3D view (every height)…" on the Wind rose tab
  stacks every boom's rose (and the mesonet's at 10 m) in a GL window, one radial scale for all.
  - It follows the tab's filtering, sectors, interval and lines of constant bearing. Its own
    choices: layers spaced evenly, by height or by log height; only the slots every layer has;
    the mesonet layer; views turned, from above, from the south.
  - Every boom's wind is read once per filtering (`load_all`), about 16 s cold on oneyear.
- **Hodograph, 2026-10-26** (`viewer/hodograph.py`, `ui/hodograph_view.py`,
  `ui/inspector/hodograph_tab.py`).
  - **The vectors:** (u, v) = (east, north) "blows-toward" components of each slot's vector-mean
    wind (boom_final ws vector_mean, wd mean); the mesonet's from its slot means.
  - **Summary → Hodograph:** each boom's mean of the slots' vectors, joined bottom to top, the
    mesonet's 10 m wind a star apart. Slots: all, a stability class, each class overlaid, or the
    selection (or the rest). "only slots every boom has" (on by default) averages every height and
    the mesonet over the same slots. A group of up to 500 slots can show each slot's hodograph
    faintly. The table: N, ū, v̄, |mean|, from-bearing, mean speed, and steadiness
    (|mean vector| / mean speed). Figure export: the "hodograph" kind.
  - On oneyear the whole year's steadiness is about 0.2 at every height (the wind turns all the
    way round); one stable night's is 0.8-0.96 with height.
  - **Inspector → Hodograph:** the slot's hodograph and the mesonet's point, with the 3 slots
    either side faintly (before blue, after red). It reads boom_final and slot_final for those
    slots only (`slot_window`).
- **Derived quantities, 2026-10-26** (`viewer/derived.py`, `ui/derived_panel.py`, Controls → Derived).
  - **A definition:** a name, a unit, variables and a formula. Each variable is one of the catalog's
    quantities (not categorical, not another derived one) with its own variant, at "each boom" (the
    derived quantity's own member) or one fixed boom (pair). Formulas use the variables' names and,
    for a per-boom quantity, z (the boom's height, m); the same safe parser as the Scatter overlay
    (`compile_expression(..., x=False)`).
  - **Its kind:** per boom if an "each boom" variable is per boom (a slot quantity's value holds at
    every boom), per pair if one is per pair (the pairs every such variable has), else per slot;
    "each" variables can't mix booms and pairs.
  - **In the viewer:** the "Derived" group of every run whose catalog has the variables (others are
    greyed in the list, with the reason). `timeline.load` computes it over the slots every variable
    has; the cache key holds the definition, so an edit is computed afresh. Saving or deleting keeps
    each panel's choice (a deleted one's panel says so). Inspector clicks open no drill target.
  - **Kept** in `<home>/viewer/derived.json`, for every run.
- **Summary: the cursor in the bottom bar, 2026-10-28** (`ui/hover_readout.py`). Over any Summary
  plot the bottom bar reads "<tab>: <axis title> = <value> · …" for each axis shown: real values on
  log axes (not their logs), a tick's label where the axis is labelled (months, booms; within half a
  unit or half the spacing, else the number), a color bar's value. Every pyqtgraph view in the dock
  is watched (again on tab changes and refreshes, for plots made later); the cursor's plot is found
  from the scene items under it.
- **Stability schemes, 2026-10-28.** Controls → Stability saves what's set as a named scheme and
  loads one again (every view reclassifies), on any run. The list shows the scheme that is exactly
  what's set, or "(unsaved changes)" after an edit; Save offers that name and asks before replacing a
  different scheme of the same name; Delete removes the one shown. Kept as `SchemeStore` in
  `<home>/viewer/stability_schemes.json`. The boundaries take 5 decimals.
- **τ variants and τ schemes, 2026-10-28** (`viewer/tauvariants.py`, `ui/tau_panel.py`, Controls → τ).
  - **Variants beyond the run's three:** a fixed τ at every ladder rung (`fixed_<rung>`; 600 s is
    naive, so not repeated) and every saved τ scheme (`scheme:<name>`), in every variant menu
    (panels, Scatter z, Derived, Find, QC, Anisotropy, the anisotropy band, Stability's ζ, and the
    inspector once computed).
  - **A τ scheme:** data (excised / unexcised); a base selection (the run's own, or a what-if of
    detection and selection settings, "Copy from Summary → τ"); optional τ per stability class, each
    class taking the base, the run's selection, or a fixed rung, plus "otherwise" for unclassified.
    Classified globally (a *copy* of a stability scheme, per slot; refreshed only by Copy) or locally
    (per boom): ζ split by the heat flux's sign, or Ri of the layer around the boom (layer or
    log-profile form, see pairs at booms) split by its own sign; neutral is |x| ≤ a threshold, or
    each boom's given share closest to 0 (exact count, threshold reported). A scheme can't classify
    by another scheme's ζ (a fixed rung's is fine; it's computed first). Kept in
    `<home>/viewer/tau_schemes.json`.
  - **Computed** with the pipeline's own code: secondary's `selected_stats` and `derived`, tertiary's
    `filtering.evaluate`/`apply` (flags read for the boom's whole run) and
    `profiles.variant_profiles` (new; the τ-dependent fits of one variant). One process per boom;
    about 35 s for oneyear (10 booms), once. Written in the run's schema to
    `<home>/viewer/tau_variants/<tag>/<key>-<hash>/` (hash: definition + the run's secondary and
    tertiary stamps; ~250 MB a variant: `boom_stats` keeps only filtered rows), plus `tau_rule` (the
    class that chose each τ) and `done.json` (counts by class and τ, thresholds).
  - **Read** through `FragmentIndex.read`: a read whose filter names a viewer variant gets its rows
    (computed first if need be); one naming no variant gets every computed one. A panel showing an
    uncomputed variant says so and loads once it's done; other views wait for it in their job.
  - **Exact:** fixed 600 s reproduces naive, the run's-own scheme mrd and mrd_unexcised, and a what-if
    with the run's settings mrd (tests). Fixed while building: Summary τ's redetection left
    mrd_unexcised rows without spectra of their own (secondary's copies of mrd) at their stored
    detection; they now copy mrd's redetection.
- **Gradient products, 2026-10-28** (replacing `viewer/pairmap.py`'s pairs at single booms).
  Tertiary now stores per-boom gradients and what's built on them in `boom_final`
  (output-schema.md §5.1: dθ/dz, dθv/dz, dU/dz, shear, N², Ri_g; per flux variant φm, φh, φh_v,
  Ri_f, Km, Kh, Pr, σw/N), one set per stencil (the second suffixed `_log`, labelled "(log
  stencil)"), so they're ordinary boom quantities everywhere. The Profile tab plots N², Ri_g and Ri_f
  (no longer MOST N²) and its pair table is Ri_b alone. The τ schemes' local classifier takes Ri_g
  (saved schemes using the old "Ri_b, layer" or "Ri, log profile" now read as Ri_g). τ variants get
  the flux products too: a cross-boom pass after the per-boom build adds them to each boom's
  `boom_final` (variant cache version 2, so cached variants recompute).
- **Derived: Duplicate, 2026-10-28.** Turns the definition shown (as edited) into a new, unsaved one
  named `<name>_2` (the next free number), to rename, change and Save.
- **"all unstable" and "all stable", 2026-10-28.** Every one-class filter (Scatter, Distribution,
  Hodograph, Scales, Composite spectra, Summary → τ) also offers the two combined groups:
  strongly unstable + unstable, and stable + strongly stable (`timeline.class_choices` / `in_class`).
  They appear only when the scheme has both classes of a group (the run's own scheme might not).
- **Scatter: constrained linear, 2026-10-28.** "Linear through origin" became "constrained linear",
  y = a + b·x with the intercept a typed in (default 0). Views saved with the old name load as it
  with a = 0 (`curvefits.RENAMED`). A fit's settings are listed by their labels ("intercept 0.5").
  "Double linear through origin" likewise became "constrained double linear" (intercept D, default 0).
- **Scatter: fits on the displayed bins; RMSE alike for fits and the overlay, 2026-10-28.**
  - A fit's bins are by default the binned line's ("as displayed": its count, mean/median, log
    bins on a log x axis, and its span, the shown x's 0.5–99.5th percentiles; the points while
    it's off). Unticking gives the fit bins of its own, as before. Views saved before keep theirs.
  - Every fit and the overlay report RMSE over all the points shown where the curve is defined,
    and over the displayed bins (the curve at the bin centres; by boom height, the overlay's own
    bin values, its height varying point to point). A fit to bins of its own also reports those.
- **Scatter: relative fit errors (least squares in ln y), 2026-10-28.**
  - Why: least squares in y let the largest y decide a fit. A constrained linear φm(ζ) on log bins
    moved strongly with the upper x percentile: b = Σx(y−a)/Σx², so each bin pulls like x².
  - "errors: in y / relative (in ln y)" on each fit. In ln y it minimises Σ(ln y − ln f)² over the
    positive y, starting from the fit in y (`curvefits._in_ln_y`). Points or bins with y ≤ 0 are
    left out, and the status says how many.
  - The power law in y keeps its y²-weighted log fit; in ln y it is unweighted. The fixed-exponent
    power law in ln y is exact: ln a = mean(ln y − b·ln x).
  - Fixed per curve: exponential and the double power law are always in ln y. ζ(Ri), whose y
    changes sign, and α(Ri), fitted in pieces, are always in y.
  - Every fit and the overlay also report RMSE in ln y over the points and the displayed bins,
    where all the y scored are positive. Older views fit in y.
- **Scatter: several fits, and fitting with or without the selection, 2026-10-26.**
  - **Fit to:** every point shown, only the selected (brushed) points, or all but them. The
    selection is part of such a fit's inputs, so a new lasso marks it earlier (or refits it with
    "auto"); with nothing selected it says so.
  - **Several fits:** a list above the form (`FitSpec` options + `FitEntry` result each), each in
    its own color with a box to show it and its RMSE. "+ fit" adds a copy of the fit being edited;
    picking one loads it into the form; "remove" drops it. Fit fits the edited one and every other
    whose inputs changed; "auto" keeps them all current. The legend names each fit's options; the
    one being edited is drawn thicker. `fit_result`/`fit_inputs` are the edited one's (the overlay's
    "from fit" uses it). Saved views keep every fit's options (`fits_state`).
- **Scatter: automatic fit, 2026-10-26.** "auto" beside Fit refits (300 ms after the last change)
  whenever the fit's inputs change: the quantities or booms, the interval, axes, sides, class,
  selection, or the fit's settings. A failed automatic fit isn't retried until the inputs change;
  the status says so. The inputs now include the axes' quantities (key, variant, definition): the
  same slots can hold another quantity, which used to leave the old fit unmarked as earlier.
- **Scatter: one stability class, 2026-10-26.** "all classes" or one class of the stability
  criterion, beside "show": it filters the slots as the selection does, so the statistics, lines,
  fits, z colors and the 3D view use only that class; a fit made before is marked earlier. Kept in
  saved views (a `LateCombo`, as the classes come with the criterion).
- **Jets → Aggregate: a hodograph, 2026-10-26.** Each boom's mean wind vector over the jet slots,
  from the jet profiles (each slot's ws mean and wd mean; `hodograph.from_profiles`), with the
  mesonet's 10 m wind over the same slots as a star (the controller loads it once per run).
  - Booms with wind in fewer than half the jet slots are left out; the rest are averaged over the
    slots they all have.
  - Faintly: each event's mean hodograph (all events), or each slot's (one event, slot by slot).
- **Scatter: a function overlay, 2026-10-26** (`viewer/expression.py`, `ui/function_overlay.py`).
  The "overlay" tab under the plot (beside "curve fit") draws y(x) typed by hand, not fitted.
  - **Formulas:** numbers, x, + - * / ^ (or **), brackets, pi, e and exp, ln/log, log10, sqrt,
    abs, tanh, sin, cos, tan, arctan, sinh, cosh. The text is parsed to a syntax tree and checked
    node by node, so only arithmetic runs; what can't be read is marked, with the reason.
  - **Pieces:** each over from ≤ x < to (blank: unbounded); where pieces overlap the first one
    listed wins; the line breaks between pieces.
  - **"from fit"** replaces the pieces with the current fit's formulas (`curvefits.fit_formula`):
    ζ(Ri) gives two pieces, α-Ri three. Every fit's formulas redraw its own curve (tested).
  - The overlay is in the values plotted (as the fits are), shows its RMSE over the points where
    it's defined, and is kept in saved views (`get_state`/`put_state` hooks in `view_state`).
  - **Boom heights, 2026-10-26:** formulas may use zA and zB, the heights (m) of the booms Scatter
    shows from panel A and panel B (x's boom where both axes use one panel). By boom height (z), each
    point uses its own boom's height: one curve per boom in its color, the RMSE against each point's
    own. A formula using one that isn't a single boom (a pair or slot quantity, or no axis from that
    panel) says so instead of drawing. `compile_expression(text, variables)` takes the names.
- **Numbers tab, all tables, 2026-10-15:** the table list starts with "all tables", so one
  filter searches every stored table of the slot.
  - It lists one value per line: `table`, `key` (the columns that pick the row, e.g.
    `variable=ts, stat=skew`), `field` (the value's column) and `value`. `slotdata.all_rows`
    builds it. slot and slot_start are left out, and boom is left out where every row is this
    boom's. A filter_log row, which has no value columns, is one line with an empty field.
  - Double-clicking a line opens its row in its own table, selected. The filter stays unless it
    would hide that row, as a key-only filter like `stat=skew` would.
  - Every table now opens in its stored order rather than sorted on the first column, and cells
    can't be edited.
- **More distribution fits, 2026-10-15:** the Distributions "fit" menu adds Rayleigh,
  exponential and beta (`distfits.py`).
  - Rayleigh and exponential have loc fixed at 0 and need every value positive. Each fits one
    parameter, its scale: σ for the Rayleigh, β (the mean) for the exponential.
  - Beta is the standard beta on (0, 1), with loc 0 and scale 1 fixed. It fits α and β and
    needs every value strictly between 0 and 1.
  - The menu now offers every fit. One the values can't take shows a note saying what it needs.
  - **Same day, second round:**
    - Chi, chi-squared and noncentral chi-squared are added, each with loc 0 and a free scale.
      Uniform is added too: a and b are the lowest and highest values.
      - Chi-squared with a free scale is the gamma family again (k = 2 × gamma's k), so it gets
        the same AIC. It's offered for its degrees-of-freedom reading.
      - Noncentral chi-squared is slow (about 1 s per 10k values). It estimates from at most 10k
        values per boom, starting from the moments. Its KS and AIC use the same ≤50k values as
        every other fit, so the AICs still compare.
    - The summary table adds skew and kurt: population skewness, and kurtosis with a normal at
      3, as the pipeline's `math.stats.moments` computes them. Directions leave them blank.
    - Only the column labels have tooltips: a few words on that column ("gamma fit: shape",
      "Akaike information criterion: lower fits better"). The cells and the table have none.
    - The "cumulative" box switches the histograms to each boom's empirical CDF, with the fit's
      CDF dashed over it.
      - Each CDF uses all of the boom's values. The range only frames the view, and the bin
        count is greyed out.
      - With slots brushed, the selection's line is highlighted in the selection colour.
      - Directions run from 0° to 360°.
      - Switching views reuses the fit: a fit reruns only when its values or distribution change.
      - Figure export draws the cumulative view too (`cdfs`, and fits with `cdf`).
    - **2026-10-16:** the summary table adds min, p25, p75 and max. The columns are now N,
      NaN %, mean, σ, skew, kurt, then the quantiles in order: min, p5, p25, median, p75, p95,
      max. Median moved there from beside the mean. Directions leave all the quantiles blank.
    - **Formulas** (Elliott chose tooltips from mockups: a typeset line, a rich-text line, a box
      in the plot, or tooltips).
      - Hovering the fit menu, or an entry in its list, shows that distribution's PDF and what
        each parameter is.
      - Hovering a boom's row label shows the PDF with that boom's fitted numbers filled in. For
        von Mises it's per degree, as drawn.
      - The rich text comes from `distfits.formula` and `distfits.filled`. A check against the
        fitted PDFs agreed to within the 4-significant-figure rounding.
    - **Textbook symbols** replace scipy's names in the table:
      - log-normal: μ and σ of ln x
      - gamma: k and θ
      - exponential: β
      - Rayleigh: σ
      - chi: k and σ (each component's σ)
      - chi-squared and noncentral chi-squared: k, λ and σ² (each component's variance)
      - beta: α and β
      - uniform: a and b
      - von Mises: κ and μ (degrees)

      A symbol that's also a summary column gets " (fit)", e.g. "σ (fit)"; its label's tooltip
      says what it is.
- **The low-level jet tool** lives in its own Jets tabs; see `docs/llj/llj-plan.md`.
- **Scatter tab:** with slots brushed, "show: all slots / selection only" picks the points.
  The statistics, lines and fits use only the shown points. A "ring the selection" check box,
  on by default, turns the rings off. Both are greyed out while nothing is brushed.
- **Distributions tab:** with slots brushed, "show" picks what's plotted.
  - **Selection and the rest** is the default, as before: the selection filled, the rest as
    lines, two tables.
  - **Selection only:** the selection alone, one table.
  - **All slots:** every slot, as if nothing were brushed.
  - It works for categories too, where "selection and the rest" shows all the slots. Each
    part's fits follow what's shown.
- **Distribution bins:** each Distributions panel has a bins row.
  - The bin count defaults to 60.
  - "Auto range" (on by default) spans the 0.1st–99.9th percentiles, or 0–360° for
    directions. While it's on, its from/to boxes show the range it chose.
  - Unticked, "from" and "to" set the range, starting from the automatic one. Values outside
    it are left out.
  - The log-bins box works with either; log bins need a lower end above 0.
- **Panel buttons:** "Controls" and "Summary" toolbar buttons (each dock's toggle action) show
  or hide those panels, docked or detached. A detached panel that was closed can be brought
  back, and pressing a button while its panel is open closes it.
- **Stability tab (Controls):** chooses how stability is classified for every view that uses
  it: the band, the Distributions, Profiles, Scatter, Spectra and Anisotropy tabs' class
  choices, and Find's `stability`.
  - **Criterion:** Ri_b between any two booms, or ζ = z/L at one boom (with its variant).
  - **Classes:** the five classes, split by four editable boundaries. Each class runs from its
    lower boundary (included) to the next (excluded).
  - **Default:** the run's own `post.stability`, converted where its classes are the five
    names as contiguous `[lo, hi)` intervals. Otherwise the run's own is used and can't be
    shown as boundaries.
  - **ζ boundaries:** they start at ±0.05 / ±0.5 (`stability.ZETA_EDGES`), only a starting point.
  - **Saving and reset:** a criterion of one's own is remembered per run (QSettings
    `stability/<tag>`), and "The run's own" goes back. The tab lists slots per class over the
    whole period, and the band's legend names the criterion.
  - **Code:** `viewer/stability.py` (`StabilitySpec`, `from_config`, `values`, `classify`);
    `timeline.stability(run, index, spec=None)`; `ui/stability_panel.py`.
  - **The run's outputs are never changed.**

## Critical files
- **New:** `src/ttu_tower/viewer/**`, `src/ttu_tower/cli/view.py`, `src/ttu_tower/cli/runs.py`,
  `tests/unit/viewer/**`.
- **Modified (small):**
  - `pyproject.toml`
  - `io/stage.py`, `io/store.py`, `io/runs.py`
  - `primary/stage_b.py`, `primary/despike.py`, `primary/rawrung.py`, `primary/products.py`
  - `secondary/detect.py`
  - the three `*/runner.py` (config copies)
  - the acceptance tests (config-copy assertions)
- **Reused as-is:**
  - `io/load.load_boom`, `io/rawfiles.accepted_half_hours`
  - `primary/stage_a` pieces, `stage_b.build_span`, `products.file_products`
  - `flags.FlagStore`
  - `timegrid` (`slot_to_time`, `majority_block`, `EPOCH`)
  - `math/polar`
  - `secondary/select.select`
  - `post/classify.stability_classes`
  - `config.load.config_from_dict` / `to_resolved_dict`, `config.hashing`
  - `io.stage.read_run_meta`
  - `validation/synthetic.write_raw_dataset` (tests)
- **Persistence:** this file (`docs/viewer/viewer-plan.md`) is the tracked copy of the plan, so a
  fresh implementation session can follow it.

## Verification
- Per phase: `pytest tests/unit` (plus `--slow` where fixtures are slow) stays green. The trace
  hooks are proven output-neutral, and the viewer tests above pass offscreen.
- The real-data integration test (`pytest --integration tests/unit/viewer`) passes against
  `oneyear`.
- Manual checkpoint with Elliott at the end of each phase, as listed. Performance targets:
  - quantity switch under 2 s cold
  - smooth pan/zoom over the full year with 10 booms
  - series view under 3 s cold and under 1 s warm
  - inspector stored-table tabs under 0.5 s
