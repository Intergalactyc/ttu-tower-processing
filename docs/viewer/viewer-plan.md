# `ttu-view`: architecture & implementation plan

## Status (resume here)
**2026-09-28:** phases 1 and 2 are built, checkpoint-tested by Elliott, and revised per his
feedback. See "Phase 1 checkpoint revisions" and "Phase 2 as built" below.

**Phases 3 and 4 are built; Elliott hasn't reviewed either yet.**
- Phase 3 includes the anisotropy band/map, the scatter tab and distribution fits, plus his
  same-day revisions (scatter interval + Fit button, the wind rose tab). See "Phase 3 as built".
- Phase 4 (QC what-if) was started at his request before his phase-3 review. See "Phase 4 as
  built".

**Next: Elliott reviews phases 3 and 4.** Then phase 5 (linked brushing, find, bookmarks);
phase 6 is the rest. Neither phase 3 nor phase 4 is committed.

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
    - the curve-fit catalogue of `curve_fits`: linear, through origin, power, log, exponential,
      neutral log law, double linear (± through origin), ζ(Ri) and α(Ri). Each fit takes
      x-percentile limits and optional binning (count, mean/median, minimum per bin).
  - Statistics: N, Pearson r, Spearman ρ, and Cohen's κ on signs.
  - The view frames the 1st–99th percentiles.
  - A click on a point opens the inspector.
- **Distribution fits** (`distfits.py`) in each histogram's "fit" menu:
  - normal, log-normal, Weibull and gamma (positive-only where needed), or von Mises for
    directions
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

### Phase 6 — Later
- Composite spectra over the visible range or a stability class.
- Two-run comparison.
- Diurnal × month heatmap and wind rose.
- Barycentric anisotropy map.
- Publication export (matplotlib).
- PSD, once the pipeline computes it, in the reserved Spectra switch.
- (Distribution fits were built in phase 3.)

---

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
