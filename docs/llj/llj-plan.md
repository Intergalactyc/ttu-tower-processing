# Low-level jet tool: plan

## Status

**All three phases (J1 the detection core, J2 controls, events and band, J3 the Event,
Aggregate and Diagnostics views) built 2026-10-01; awaiting Elliott's checkpoint.**
- The detection rules and their defaults are in `llj-detection.md`. That document is the
  specification for the detection code.

## Goal

Find possible low-level jets in the tower data and explore them in ttu-view:
1. **Detection algorithms with one common signature**, so they can be swapped, compared,
   extended and used outside the GUI. They live in a new package, `ttu_tower.jets`.
2. **A jets tool inside ttu-view** that:
   - runs any registered algorithm over the data, with its parameters tunable
   - filters the resulting events by time of day, night, inversion strength and Richardson
     number
   - shows individual jets (their profiles and statistics) and the jets in aggregate

Results are explored in the GUI only for now. The data model (plain DataFrames, parameters that
serialize to JSON) leaves room for export later.

## Decisions (Elliott, 2026-10-01)

- **Wind speed:** the 10-min slot means (`boom_final`, `ws mean`, variant `none`).
- **Slot length:** always 10 min, so counts in slots are counts of time (6 slots = 1 h).
- **First algorithm:** a corrected, generalized version of the old discrete method:
  - **Nose:** a strict local maximum, never the top boom, with the falloff measured above it.
  - **Required booms:** the high booms it needs must all be present and good.
  - **Thresholds:** an absolute (default 2 m/s) and/or relative falloff threshold, and a
    10 m/s minimum nose speed.
  - **Defaults kept from the old pipeline:** the booms used, a 1-slot gap fill and a 1-h
    minimum event.
  - The interface stays open for more algorithms, chosen in the GUI.
- **Where the code lives:** algorithms in `src/ttu_tower/jets/identify.py`; the GUI as new tabs
  in ttu-view.
- **Filters in the GUI:**
  - time of day: a clock window, plus a night-only check box (`slot_final.night`)
  - inversion strength: ΔVPT between two booms, default 2 and 10
  - Ri_b between two booms, default 2 and 4
- **No export yet.**

---

## Part A — What the user sees

### A1. The Jets tab in the left dock (controls)

Next to Find and Bookmarks. The result counts shown are the prototype's on `oneyear` under the
defaults:

```
┌ Jets ──────────────────────────────────┐
│ Algorithm  [Peak (local max + falloff) ▾]
│ ── parameters (built from the algorithm) ──
│ profile booms   [1 2 3 4 5 6 7 8 9 10]
│ required booms  [7 8 9 10]
│ min peak speed  [10.0] m/s
│ ☑ min falloff   [ 2.0] m/s
│ ☐ min falloff   [0.25] of peak speed
│ ── events ──
│ bridge gaps up to [1] slots   min length [6] slots
│ ── filters ──
│ ☐ time of day  [18:00]–[09:00]   ☐ night only
│ ☐ ΔVPT(b[10] − b[2]) ≥ [  1.0] K
│ ☐ Ri_b(b[2], b[4]) in [    ]–[ 0.25]
│ ── result ──
│ slots with all required booms   26,739
│ no peak 12,772 · too slow 10,149 · weak falloff 3,477
│ jet slots 341 → 20 events → N after filters
│ [Select jet slots] [Step through events] [Reset]
└────────────────────────────────────────┘
```

- **Live rerun.** Any change reruns detection after a short pause (debounced). It's fast enough
  to feel live: arrays are loaded once per run, and detection is vectorized.
- **The result funnel.** It counts slots by `status` (`llj-detection.md` §3.3), then jet
  slots, events and events left after the filters. It shows at a glance which threshold is
  doing the work.
- **Select jet slots** puts the jet slots, or the filtered events' slots, into the brushed
  selection. Every Summary tab can then already compare jets with the rest: distributions,
  scatter, the profiles tab's "brushed selection | the rest", composites and the rest.
- **Step through events** opens the Slot Inspector on each event's fastest slot, with the
  existing ◀ match / match ▶ stepping.

### A2. The jets band (on the timeline)

A thin row under the stability and anisotropy bands. Each event is a span, coloured by nose
height and hatched for events a filter removed. Hovering shows the event's start, duration and
nose. A click selects the event in the Jets view; a double-click opens the Slot Inspector. A
check box in View shows or hides the band, like the existing bands.

### A3. The Jets tab in the right (Summary) dock

It has four sub-tabs.

1. **Events.** A sortable table, one row per event:
   - id, start and end, duration
   - nose height (median, range) and nose speed (max)
   - falloff (max), nose direction
   - ΔVPT and Ri_b (the filter quantities), filled slots, truncated
   - whether the filters passed it

   Selecting a row opens it in **Event**, and the timelines jump to it.
2. **Event.** One jet, padded by a margin of slots before and after:
   - **Time–height speed map:** booms × slots, with the noses marked and the event's span
     outlined.
   - **Profiles:** ws against height for each slot, coloured by time, noses marked. Optional
     panels show wd and VPT against height.
   - **Event statistics:** the event's table row, plus its slot list with each slot's
     `status`.
3. **Aggregate.** Over the filtered events, or all of them:
   - **Histograms:** nose height (as bars per boom, since heights are discrete), nose speed,
     falloff, duration and start hour.
   - **Counts by hour × month:** uses the diurnal heatmap code.
   - **A composite profile:** the median ws profile of jet slots with its spread, optionally
     normalized by nose speed and height.
   - **A rose of nose directions:** uses the wind-rose code.
4. **Diagnostics.** How the thresholds act:
   - histograms of candidate nose speed and falloff, with the threshold lines
   - jet-slot and event counts as each threshold sweeps across its range, the others held
     fixed
   - the status table

A toggle shared by Aggregate and Diagnostics chooses between **whole period** and **visible
range**, like the Distributions tab.

**Figure export:** the existing "Export figure…" and right-click export work on all of these
plots unchanged.

---

## Part B — Architecture

### B1. Package layout

```
src/ttu_tower/jets/              Qt-free; usable from notebooks and a later export
  __init__.py
  identify.py   Profiles, the parameter metadata helpers, the registry, algorithm "nose"
  events.py     gap filling, minimum length, grouping, event statistics
  filters.py    time-of-day, ΔVPT and Ri_b event filters
  data.py       loading Profiles and the filter quantities from a run's tertiary output
  analysis.py   JetSettings (serializable) and analyse(): detection -> events -> filters -> counts

src/ttu_tower/viewer/ui/
  param_form.py   a form generated from a parameter dataclass's metadata
  jets_panel.py   the left-dock Jets tab
  jets_view.py    the right-dock Jets tab (Events, plus the sub-tabs below)
  jets_event.py   the Event sub-tab
  jets_stats.py   the Aggregate and Diagnostics sub-tabs
  jets_controller.py  loading, live rerun, and the hooks into the main window
  overview.py     + JetsBand next to StabilityBand / AnisotropyBand
  main_window.py  wiring: dock tabs, band, selection, inspector stepping, jobs
```

### B2. `jets/identify.py`

- **`Profiles`, the output columns and the status codes**, exactly as in
  `llj-detection.md` §2.
- **`param(default, *, label, unit, kind, lo, hi, help)`:** wraps
  `dataclasses.field(metadata=…)`, so each algorithm's parameter dataclass describes itself.
  - Kinds: `float`, `optional_float` (an enable check box plus a value), `int`, `booms`.
  - Parameter sets are validated before running, for example "at least one falloff threshold
    on" or "required booms ⊂ profile booms". The GUI shows the validation errors.
  - `to_dict` and `from_dict` serialize a parameter set to and from JSON.
- **The registry:** a `register(key, label, params)` decorator and `ALGORITHMS`. A new
  algorithm is one decorated function plus its parameter dataclass. The GUI lists whatever is
  registered.
- **`nose`:** vectorized over slots and loops only over the ~10 booms.
  - It computes each candidate boom's local-max mask, its minimum-speed-above (a reverse
    cumulative `nanmin`), falloff, speed test and falloff test.
  - It then picks the qualifying candidate per slot and assigns status codes in rule order.

### B3. `jets/events.py`

- `fill_gaps(is_jet, max_gap) -> (jet, filled)`
- `drop_short(jet, min_slots)`
- `group(slots, jet, filled) -> events`
- `event_stats(events, detection, context)`

All are run-length based (`np.diff` of the mask), with no scipy morphology. Rules are in
`llj-detection.md` §4.

### B4. `jets/filters.py`

- `time_of_day(events, start, end, tz)`, which handles windows that wrap midnight
- `night_only(events)`, from the start slot's `slot_final.night`
- `inversion(events, threshold)`
- `richardson(events, lo, hi)`

Each returns a boolean mask over events. A `JetFilters` dataclass holds the enabled filters and
their settings, and is serializable like a parameter set.

### B5. `jets/data.py`

- `load_profiles(run_dir, booms=BOOMS) -> Profiles`
  - Reads tertiary `boom_final` (`ws`, `mean`, `none`) for the booms through
    `io.store.read_table` with a pushdown filter.
  - Pivots to slots × booms and reindexes to a complete slot grid.
  - Adds `wd` (`mean`, `none`) as an extra array, for the nose direction.
- `load_context(run_dir, vpt_booms, rib_pairs) -> DataFrame` reads the per-slot `vpt` for the
  chosen booms, `rib` for the chosen pairs, and `slot_final.night`. It's loaded only when a filter or statistic
  needs it, and cached per (run, boom set).
- On `oneyear`, loading all 10 booms of ws and vpt together took 1.2 s.

### B6. Viewer side

- **Background work:** loading goes through the existing `JobRunner`, with the results cached
  on the `RunHandle`. Detection, events and filters run on the GUI thread; they're
  milliseconds, so the form feels live without job-queue latency. Timing is to be checked;
  if detection turns out slow, it becomes a keyed, superseding job like everything else.
- **Reuse:**
  - the brushed selection (`MainWindow.set_selection`)
  - inspector stepping (`inspect_sequence`)
  - the diurnal grid (`viewer.diurnal`) and wind-rose code
  - the band pattern (`StabilityBand`)
  - `tables.number` formatting, figure export and the right-click CSV export
- **Settings:** the algorithm, parameters and filters are remembered per user (QSettings, as
  JSON), so the tool reopens as it was left.

### B7. Tests

- **`tests/unit/jets/test_jets_identify.py`:** built from hand-made `Profiles`:
  - a top-boom peak is rejected
  - a strict local maximum is needed, and ties go to the lower boom
  - a missing neighbour drops the candidate
  - required booms are enforced
  - absolute, relative and combined falloff
  - minimum speed is inclusive
  - when several candidates qualify, the fastest wins
  - each status code is reported, and the reported candidate is right for each
  - registry and parameter metadata round-trip; invalid parameter sets are rejected
- **`test_jets_events.py`:**
  - gap filling (including gaps at the ends and missing slots)
  - minimum length, counted after filling
  - edges and truncation
  - statistics come from detected slots only
  - circular mean of direction
- **`test_jets_filters.py`:** windows that wrap midnight, both time zones, NaN statistics,
  open-ended Ri ranges.
- **`test_jets_data.py`:** on small tertiary tables written by the test (two fragments, a slot
  with no rows, other variants and stats present), `load_profiles` and `load_context` read
  exactly the right rows onto a complete slot grid.
- **Viewer (slow; `test_viewer_jets_ui.py`):**
  - the form is generated from the registry, and changing it reruns detection
  - the funnel counts match the detection frame
  - selecting events brushes their slots
  - a band click opens the event; a double-click opens the inspector
  - the synthetic fixture has no jets, so these tests inject a jet-shaped `Profiles`
- **Integration (`--integration`, `test_jets_integration.py`):** on `oneyear`:
  - the vectorized detection agrees with a slot-by-slot reading of the rules
    (`jets_reference.py`) on every slot, for two parameter sets
  - events obey their own rules
  - loading takes under 5 s and detection under 0.5 s

All new test basenames will be checked for uniqueness across `tests/`.

---

## Part C — Phases (Elliott tests after each)

**Phase J1 — the detection core.**
- `jets/identify.py`, `events.py`, `filters.py`, `data.py` and their tests.
- **Checkpoint:** the counts on `oneyear` under the defaults and a few variants.

**Phase J2 — controls, events and the timeline.**
- The left-dock Jets tab: the generated form, live rerun, the funnel, filters, Select and Step
  through.
- The Events sub-tab, the jets band, and the inspector and selection wiring.

**Phase J3 — looking at jets.**
- The Event, Aggregate and Diagnostics sub-tabs; the whole-period / visible-range toggle.

### Phase J1 as built (2026-10-01)

- **`ttu_tower/jets/identify.py`:**
  - `Profiles` (with `from_frame` and `subset`) and the parameter helpers: `param`, `Params`,
    `describe`, `to_dict`, `from_dict`
  - the registry: `register`, `ALGORITHMS`, and `detect(key, profiles, params)`, which checks
    the parameters and the output contract
  - `result_frame`, a helper for building the contract's columns
  - the `nose` algorithm with `NoseParams`
- **`ttu_tower/jets/events.py`:** `EventParams`, `fill_gaps`, `find_events -> Events(table,
  slots)` and `detected_slots`.
- **`ttu_tower/jets/filters.py`:**
  - `JetFilters` (validate, enabled, to/from dict) and `context_needs`
  - `event_quantities`, which gives the start minute, the start night flag, the mean ΔVPT and
    the median Ri_b
  - `apply_filters`, which gives one column per filter plus `passed`
- **`ttu_tower/jets/data.py`:**
  - `load_profiles` reads ws and wd, `none` variant mean, onto the full slot grid.
  - `load_context` reads `vpt_b{b}`, `rib_b{a}-b{b}` and `night`.
- **Tests:** `tests/unit/jets/`, 41 tests including the 4 integration ones.
  - `test_jets_identify.py`, `test_jets_events.py`, `test_jets_filters.py`, `test_jets_data.py`
    and `test_jets_integration.py`, plus the helper `jets_reference.py`
  - The randomized and real-data checks compare against a slot-by-slot reading of the rules.
- **On `oneyear`:**
  - Loading takes 1.5 s for the profiles and 2.2 s for the filter context. Detection takes
    0.03 s and events 0.01 s, so J2 can rerun everything on the GUI thread as planned.
  - The defaults give 341 jet slots in 20 events; 11 pass night only plus ΔVPT ≥ 1 K
    (`llj-detection.md` §6).

### Phase J2 as built (2026-10-01)

- **`ttu_tower/jets/analysis.py`:**
  - **`JetSettings`:** the algorithm, its parameters, the event parameters and the filters.
    - `to_dict` keeps other algorithms' saved parameters.
    - `from_dict` falls back to defaults on stale or broken saves.
  - **`analyse(profiles, settings, context, tz) -> JetResult`:**
    - `JetResult` holds the detection, events, filter quantities and verdicts.
    - `counts()` gives slots per status, jet slots, events, events kept, and per filter how
      many events it drops and how many of those for lack of a value.
    - `event_slots()` gives the slots of chosen events.
  - Events gained `peak_slot` and `peak_boom`.
- **`viewer/ui/param_form.py`:** a `ParamForm` built from `describe()`.
  - Booms are toggle rows; optional floats have an enable check box, which restores the
    suggested value.
  - Labels move above their fields when the dock is narrow.
- **`viewer/ui/jets_panel.py`:** the left-dock **Jets** tab.
  - The algorithm and its description, the generated detection form, the events form, and the
    filters (time window, night only, ΔVPT with its booms, Ri_b with its booms and optional
    minimum and maximum).
  - Validation errors are shown in red, and the last good result stays.
  - The result counts, plus **Select jet slots**, **Step through events** and **Defaults**.
- **`viewer/ui/jets_view.py`:** the right-dock **Jets** tab, with the **Events** sub-tab.
  - A sortable table with one row per event: start, length, nose, maximum speed, maximum
    falloff, nose direction, ΔVPT, Ri_b, bridged slots, and the filter verdict.
  - Header tooltips name the boom pairs. "Only events the filters keep" is on by default;
    removed events are listed grey when it's off.
- **`overview.JetsBand`:** events as spans in the colour of their nose's nearest boom. Events the
  filters remove are drawn **grey**; the plan said hatched, but grey reads more clearly at a
  year's zoom. The chosen event is outlined, hovering names an event, a click selects it and a
  double click steps through events from it.
- **`viewer/ui/jets_controller.py`:**
  - **Loading:** on opening a run, it loads the profiles and the filter quantities as
    background jobs. The filter quantities reload when the filter booms change.
  - **Rerunning:** the detection reruns on the GUI thread 150 ms after any change
    (milliseconds on `oneyear`), and again when the time zone changes.
  - **Saving:** the settings go to QSettings as JSON (`jets/settings`).
  - **Picking an event** sets the timelines to it, padded by at least 3 h either side.
  - **Stepping through events** opens the Slot Inspector's Profile tab on each kept event's
    `peak_slot` at its `peak_boom`, as a match sequence. `MainWindow.inspect_sequence` gained
    a `tab` argument for this.
- **Main window:**
  - the Jets tabs in both docks
  - the band under the anisotropy band, with a "jets band" check box
  - the axis-width sync and the time-zone hook
- **Tests:**
  - `tests/unit/jets/test_jets_analysis.py` (4)
  - `tests/unit/viewer/test_viewer_jets_ui.py` (6, slow): generated form, live rerun and
    validation, picking from the table and the band, Select, Step through, and saved
    settings plus time zone. The synthetic run has only booms 1–2 and no jets, so these
    tests inject jet-shaped profiles that nose at boom 2.

### Phase J3 as built (2026-10-01)

- **Event sub-tab** (`viewer/ui/jets_event.py`):
  - **Header:** ◀ ▶ steps between events (only kept ones while the Events list shows only
    those), and the event's description.
  - **Time–height map:** wind speed for every boom over the event, with at least 3 h either
    side. The colour bar is capped at the window's 99th percentile, and missing data is
    blank. Stars mark each detected slot's nose, crosses mark bridged slots along the top, and
    the event is outlined. Double-clicking a cell opens that slot and boom in the Slot
    Inspector's Profile tab.
  - **Profiles:** wind speed against height, one curve per slot coloured by time
    (purple → yellow, keyed above the plots), dashed for bridged slots, noses starred.
  - **Optional direction and VPT profiles:** VPT for every boom is loaded only when first
    asked for.
  - **Slot table:** each slot's time, status (bridged noted), nose, speed, falloff and the
    height of the slowest boom above.
- **Aggregate sub-tab** (`viewer/ui/jets_stats.py`): the events the filters keep, or all of
  them.
  - **Histograms and bars:** nose height (each event's median, at the nearest boom), maximum
    nose speed, maximum falloff, duration and start hour.
  - **Hour × month heatmap:** the share of slots with data that are detected jet slots of
    those events, built with `viewer.diurnal.grid`.
  - **Composite profile:** the median and interquartile band of the jet slots' wind
    profiles, optionally relative to each slot's nose speed.
  - **Rose:** nose direction and speed, using the wind-rose code.
- **Diagnostics sub-tab:**
  - **Status counts:** slots per status, as a count and a percentage.
  - **Candidate histograms:** each slot's reported candidate's nose speed, falloff and
    relative falloff. Jet slots are overlaid in red, the active thresholds are dashed, and
    the counts can go on a log scale.
  - **Sweeps:** any numeric setting (the algorithm's or the events') swept over a range
    around its current value. The jet slots, events and kept events are plotted against it,
    with the current value dashed. A threshold that is off is switched on for the sweep.
  - Sweeps run as a background job (`analysis.sweep`; an event setting reuses one
    detection), rerun when the settings change, and cost about 1 s on `oneyear`.
- **Whole period or visible range:**
  - **Aggregate and Diagnostics** follow the Summary dock's shared "whole period" check box,
    like the other summary tabs: counts cover slots in the range, and events must start in
    it.
  - **The Events list** always shows every event.
- **`ttu_tower/jets/analysis.py`:**
  - `JetResult.counts(slot_range)`, `events_in(slot_range, passed_only)`, `detected_in(ids)`
  - `sweep(...)` and `sweep_values(field, current)`
- **Not done: matplotlib figures.** The toolbar's "Export figure…" has no figure for the
  Jets tab yet and says there's nothing to draw. Each plot's right-click export, including
  the labelled CSV, works.
- **Tests:**
  - `test_jets_analysis.py`: three more tests, for range counts, sweeps and sweep values.
  - `test_viewer_jets_ui.py`: five more slow tests:
    - the Event map, its slots and event stepping
    - a double click on the map opening the inspector
    - VPT loading
    - the Aggregate following the interval
    - the Diagnostics status and sweeps

### Manual edits (Elliott, 2026-10-02)

Events can be edited by hand. The rules are in `llj-detection.md` §4.1.

**Elliott's choices:**
- Edits are saved per run, clearly labelled as manual, and easy to remove, some or all.
- An edited event whose detected event disappears stays as a manual event.
- Rejecting an event and adding one by hand are included.
- A switch chooses whether nose statistics come from jet slots only or from every slot of the
  event.

**What was built:**
- **`ttu_tower/jets/edits.py`:**
  - `Edit` and `new_edit`
  - `EditStore`, one JSON file per run under `<home>/viewer/jet_edits/`
  - `apply_edits(events, detection, edits, ...)`, which returns the new `Events` and a report
- **`events.py`:**
  - `assemble` builds an event table from any set of spans.
  - Events gain `source`, `rejected`, `edits`, `n_manual` and a per-slot `manual` flag.
  - `nose_stats` and `nose_rows` take `nose_from` ("jets" or "candidates").
- **`analysis.analyse(..., edits=)`:**
  - It applies the edits after grouping.
  - Rejected events fail `passed`, and `counts()` adds `rejected`, `edited` and `manual`.
  - `JetResult.table(nose_from)` and `slots_in(ids)` are new.
  - Sweeps ignore edits; they're about the automatic detection.
- **Event sub-tab:**
  - **Edit boundaries:** the map widens to 12 h either side, and the event's span is dragged
    (snapping to slot edges) and saved.
  - **Reject / Rejected (undo).**
  - **Undo manual edits:** removes every edit of the event.
  - The header names the event's source; hand-added slots are dashed and labelled in the slot
    table.
- **Events sub-tab:**
  - A "source" column: detected, edited or by hand. Edited and hand-made rows are tinted and
    italic.
  - The "kept" column says "rejected by hand" where it applies.
  - **Add event from selection:** an event from the first to the last brushed slot.
  - **Manual edits (n)…:** a dialog listing every edit (kind, span, event, what it does now,
    when made). It can show an edit's event and remove the selected edits or all of them.
- **Jets tab switch:** "every slot of an event is a jet slot" sets the nose statistics in the
  Events, Event and Aggregate sub-tabs.
- **Jets band:** an orange strip along the top marks edited and hand-made events. Rejected
  events are dark red, and hand-made events without a nose are dark grey.
- **Filter quantities now use the whole span:** ΔVPT and Ri_b use every slot of an event,
  where before they used only its detected slots.
- **Tests:**
  - `tests/unit/jets/test_jets_edits.py` (9)
  - three more slow UI tests in `test_viewer_jets_ui.py`: dragging boundaries, which saves,
    replaces and undoes; rejecting and undoing; adding from the selection and the edits dialog

### Review changes (Elliott, 2026-10-02)

- **Event sub-tab, two profile panels:** they replace the direction and VPT check boxes.
  - Each panel has a dropdown: wind speed, VPT, direction, TI, TKE, σw, wd std, w'u', w'v',
    w'θv' or ζ = z/L (ζ added later the same day). The defaults are wind speed and VPT.
  - The second-moment quantities are the selected-τ (`mrd`) values. Each loads on first use
    (`jets.data.load_boom_values`).
- **Aggregate sub-tab:**
  - **Units:** the histograms are densities, so each integrates to 1. The nose-height and
    start-hour bars are fractions of the events.
  - **Ranges:** every plot refits its axes on each redraw: a new event selection, the "whole
    period" box, or the composite's "relative to the nose speed" switch.
- **Diagnostics sub-tab:** only the slot counts and the sweep remain; the candidate histograms
  are gone. The sweep note says sweeps ignore manual edits.
- **Second round:**
  - **Profile panel modes:** each Event panel has a mode: every slot, mean ± SD or median and
    IQR, over every slot of the event.
    - Direction is summarized about its circular mean and unwrapped up the profile, so it
      doesn't jump at north.
    - Both panels refit their axes on every redraw.
  - **One event slot by slot:** the Aggregate sub-tab can show the picked event alone. Its
    nose height, nose speed, falloff and hour are distributed over its slots, along with its
    composite profile and rose. The duration and heatmap plots say they need all events.
  - **Select event slots:** a button on the Event sub-tab and under the Events list brushes
    every slot of the shown or selected event, bridged and hand-added ones included, into the
    shared selection.

### Observed peak and estimated nose (Elliott, 2026-10-02)

The rules are in `llj-detection.md` §4.2.
- **New module:** `ttu_tower/jets/noses.py` (`vertex` and `estimate`). `analyse` attaches the
  estimate to the detection, and `events.nose_stats` adds the `z_est_*`/`u_est_*` columns.
- **GUI wording:** "nose" (the fastest boom) is now the **observed peak**, and the fitted maximum
  is the **estimated nose**.
  - **Events table:** obs. peak (m), est. nose (m), obs. peak speed (m/s), est. nose speed
    (m/s). The tooltips give their ranges.
  - **Event sub-tab:** the header gives both. The slot table lists both per slot. The wind
    speed panel marks observed peaks with stars and estimated noses with red diamonds.
  - **Aggregate sub-tab:** a switch between "observed peaks" and "estimated noses".
    - Observed heights are bars per boom; estimated heights form a 10-m density histogram.
    - It also sets the speed histogram and what the relative composite divides by.
  - **Jets band legend:** "peak at … m".
- **Detection settings say "peak" too:** the algorithm is "Peak (local maximum with
  falloff)", with "min peak speed" and a relative falloff "of peak speed".
  - The internal key `nose`, the parameter names (`min_nose_speed`, …) and the detection
    columns (`z_nose`/`u_nose`) are unchanged, so saved settings still load.
- **Tests:** `tests/unit/jets/test_jets_noses.py` (5) and one more slow UI test.

**Later, when wanted:**
- matplotlib figures of the Jets views ("Export figure…")
- exporting events and parameter sets
- more algorithms, such as a corrected quadratic fit or Bonner-style strength categories
- comparing two algorithms or parameter sets (overlap of events, a two-colour band)
- a wind direction criterion

---

## Resolved questions (Elliott, 2026-10-01)

1. **Strict defaults stay.** The aim is a few events that are certain, quality over quantity.
   They may be relaxed slightly later.
2. **Time of day** gets a night-only check box (`slot_final.night`, tested at the event's
   start) beside the clock window.
3. **Filter statistics:** ΔVPT is the event mean, Ri_b the event median, and time of day and
   night are tested at the event start.
4. **Missing neighbour:** a candidate nose whose neighbouring boom is missing is rejected.
5. **Gap filling** bridges only slots that have data but failed the rules, never a `no_data`
   slot. Missing data always ends an event.
   - Amended 2026-10-02: an option, `bridge_missing` (off by default), also bridges `no_data`
     slots, whether an outage or a required boom missing or flagged.
6. **Wind speed:** the plain scalar mean (`ws mean`).
