# ERA5 at the tower: plan

## Status

**Reviewed and approved by Elliott 2026-11-09. Phase E1 built 2026-11-09.**
- Downloads are running into `C:/Users/ellwalke/Data/ERA5_TTU`; logs are
  `download_sfc_pl.log` and `download_ml.log` there.
- Rerun the same command to resume or retry.
- E1 findings are recorded in A2.5, A3 and C1.

## Goal

Get ERA5 reanalysis for the tower's location for the oneyear period, and process it into:
- **profiles** at the tower heights and at fixed heights above the tower top, up to about 5 km
- **large-scale quantities** at each of those heights:
  - the geostrophic wind
  - its vertical shear (the baroclinic or thermal-wind shear), computed two ways (see A2.1)
  - the large-scale vertical velocity
- **surface fields** at the tower

The main interest is the influence of baroclinic shear and subsidence on boundary-layer
height. Cross-isobar angle is out of scope.

## Decisions (Elliott, 2026-11-09)

- **Three download commands:** surface, pressure levels and model levels.
- **A separate processing program**, which:
  - interpolates variables to the tower heights plus fixed, roughly log-spaced heights up to
    about 5 km
  - takes `z` (orography) and `fsr` (roughness) from the surface fields
  - takes the instantaneous fluxes and `blh` from the surface fields
  - computes geostrophic winds and large-scale `w` at several heights
  - estimates baroclinic shear at several heights, both from the vertical gradient of the
    geostrophic wind and from the horizontal temperature gradient
- **Avoid the Caprock Escarpment, and account properly for the sloping terrain.**
- **Later, optional:** a fourth step that combines processed ERA5 with tower data.

### Review (Elliott, 2026-11-09)

- **Stencil:** the A3 box for initial analysis. The download box is larger (C1), so the
  analysis box can be widened or moved without downloading again. (At E1 the box's east edge
  moved from 101.50 °W to 101.75 °W to stay off the escarpment; see A3.)
- **Shear:**
  - Compute the pressure-method vs temperature-method shear as a sanity check.
  - Also compute and compare the constant-height (`tz`) vs slope-corrected (`teta`) versions.
- **Accepted as written:**
  - virtual temperature throughout (A2.3)
  - ln p, not p, interpolated in height (A2.4)
  - the ratio-1.25 height ladder (A2.6)

---

## Part A — Things that shape the design

### A1. No surface geostrophic wind is needed

- **Shear doesn't need the absolute geostrophic wind.** Baroclinic shear is ∂**v**g/∂z, which
  the thermal-wind relation gets from the horizontal virtual-temperature gradient on
  constant-height surfaces.
- **The temperature route is well-conditioned near the ground; the pressure route isn't.** On
  terrain-following levels, ∇_z T = ∇_η T − (∂T/∂z)∇_η z. The two terms are of similar size,
  with no large cancellation. The pressure equivalent, ∇_z p = ∇_η p + ρg∇_η z, cancels by
  about a factor of 10 on this slope.
- **So:**
  - the geostrophic wind itself is computed only where it's clean (see A3)
  - the shear also comes from temperature, down to the lowest model level
- **Orientation:** what baroclinic shear does to BL height depends on its direction relative to
  the low-level flow (veering vs backing, i.e. warm vs cold advection). Use the actual
  near-surface wind (tower or ERA5) as the reference; that needs no surface geostrophic wind.
- **A by-product, not built:** a surface geostrophic wind could later be had by integrating the
  shear downward from the lowest clean geostrophic wind.

### A2. Flaws to be aware of

1. **In ERA5 the two shear methods aren't independent.** ERA5 is hydrostatic, so the shear of
   the pressure-gradient geostrophic wind and the temperature-gradient thermal wind are the same
   information. Where both are computed the same way, they agree up to numerics. Agreement
   checks the code, not the physics.
   - The useful comparison is the **constant-height** temperature gradient vs the
     **terrain-following-with-slope-correction** one (B4). Above the height where both exist,
     the comparison tests the slope correction that the lowest levels depend on.
   - There is no observational check of ERA5's baroclinicity from the tower.
2. **The low-level baroclinicity is where ERA5 is weakest.** Over this slope, the BL
   temperature gradient is mostly terrain heating and cooling, which is also Holton's mechanism
   for the low-level jet.
   - At night the slope term is roughly (inversion strength) × (slope).
   - ERA5's stable BL is over-mixed (inversions too weak, too deep), so expect nocturnal
     low-level shear to be **underestimated**.
   - Aloft, and in daytime, the gradients are constrained by assimilated radiosonde and aircraft
     data; in the lowest few hundred metres at night, they're largely the model's own.
   - The tower measures the inversion directly; see Part E.
3. **Use virtual temperature, not temperature.** This is dryline country. A 10 g/kg humidity
   change across 100 km is about 1.8 K/100 km of virtual-temperature gradient, the same size
   as typical thermal gradients.
4. **Interpolate ln p in height, not p.** Linear interpolation of p over ~250 m spacing biases
   it by ~0.1 hPa. The bias differs between columns at different terrain heights, which puts
   errors of order 0.5 m/s into the geostrophic wind.
5. **Model-level data is the slow part.** It comes from the tape archive, one request per day,
   probably several days in all.
   - **Format (settled at E1):** GRIB. CDS cost limits charge netCDF conversion at 6×, so a
     netCDF day (129,024) exceeds the 120,000 limit; GRIB is 21,504.
   - The pip `eccodes` and `cfgrib` packages work on Windows.
6. **Log spacing coarser than ERA5's own between 200 m and 1 km.** ERA5 levels there are
   30–80 m apart. A sparse log ladder (ratio ~1.4) puts only ~3 points across a jet nose. The
   default ladder in C2 uses ratio ~1.25; the list is configurable.
7. **For BL height, use vertical velocity relative to the terrain.** On a 1–2 m/km slope, a
   10 m/s up-slope wind is ~1–2 cm/s of vertical motion, as large as synoptic subsidence. ω
   includes it. A boundary layer measured above the ground sees w − **V**·∇z_s, so both are
   output (B5).

### A3. Terrain and the escarpment

- **One stencil box** of 0.25° grid points, around the tower, used for every horizontal
  gradient and area mean.
  - Gradients come from a least-squares plane fit over the box: a mean gradient over ~100 km,
    which is the scale at which geostrophic balance holds and ERA5 is reliable.
  - A plane fit, not finite differences: a 1 m geopotential-height error over 100 km is
    ~1.2 m/s of geostrophic wind.
- **Default (settled at E1):** 33.25–34.00 °N by 102.50–101.75 °W, which is 4 × 4 points,
  about 83 × 70 km. Its centre is ~7 km west of the tower.
  - **Why not the draft's 101.50 °W east edge:** in ERA5's orography the escarpment's drop
    starts there in the south. The 33.25 °N, 101.50 °W column is 84 m below its western
    neighbour, so the terrain plane-fit residual was 11.7 m RMS (36.7 m max). The default box
    gives 3.3 m RMS (6.7 m max).
  - **ERA5 terrain over the box:**
    - slope ∂z/∂x = −1.66 m/km, ∂z/∂y = +0.85 m/km
    - ground up to 102 m above ERA5's ground at the tower (1019 m, vs the real 1014 m)
  - The box may be asymmetric; extending it west/north/south is safe, east isn't.
- **Constant-height surfaces** (height above sea level) are the true horizontal. A height is
  valid for constant-height gradients only if it is above the lowest usable level in **every**
  box column:
  - model levels: the lowest model level
  - pressure levels: the lowest level with p < `sp`
  - With the box's highest ground 102 m above the tower's, constant-height results start
    about 115 m above the tower's ground.
  - Below that, only the slope-corrected temperature method gives shear.

---

## Part B — Processing methods

### B1. Columns

Every source, pressure or model levels, becomes the same thing: per box grid point and hour, a
column of (z_asl, p, T, q, u, v, ω).
- **Pressure levels:**
  - z = Φ/g₀, with g₀ = 9.80665.
  - Levels with p > `sp` are dropped. ERA5 extrapolates underground values; they must never be
    used.
- **Model levels:**
  - heights from the hypsometric integration below (B2)
  - pressure from half-level pressures: p = a + b·p_s
- The constant-height methods (B3, B5) work on either source. The slope-corrected method (B4)
  and the profiles (B6) need model levels.

### B2. Model-level heights

Port ECMWF's `compute_geopotential_on_ml`:
1. Half-level pressures from the L137 a/b table, read from the GRIB `pv` array.
2. Integrate R_d·T_v·d(ln p) up from the surface geopotential.
3. Full levels use the α term.

Details:
- **Use ECMWF's constants** (R_d = 287.06, g₀ = 9.80665) so the heights match ERA5's own.
- **Only levels below the top requested level are needed**; the integration runs upward from
  the ground.
- **Heights are geopotential, not geometric.** The difference at 5 km is a few metres, which is
  ignored.
- **Check:** at E3, compare the integrated heights to the pressure-level Φ at matching
  pressures.

### B3. Geostrophic wind and pressure-method shear (constant height)

1. In each box column, interpolate ln p and T_v linearly in z to each output height's z_asl.
   - z_asl = ERA5's ground height at the tower + output height above ground.
2. At each height, plane-fit ln p; then ∇p = p ∇ln p.
3. ρ = p/(R_d T_v), from the tower column's values.
4. Geostrophic wind: **v**g = (1/(ρ f)) **k** × ∇_z p.
   - f is the Coriolis parameter at the site latitude (`physics.earth.coriolis`).
   - Plane-fit coordinates: x = a·cos φ₀·Δλ, y = a·Δφ, a = 6371.229 km.
5. Pressure-method shear: centred differences of **v**g between output heights (one-sided at
   the ends).
   - Its effective vertical resolution is the source's, not the output ladder's.

### B4. Temperature-method shear (thermal wind), two versions

The exact form, on height surfaces:

∂**v**g/∂z = (g/(f T_v)) **k** × ∇_z T_v + **v**g ∂(ln T_v)/∂z

- The second term is ~5% of the first.
- Below the lowest valid height for **v**g, it takes **v**g from that lowest valid height.

The two versions differ in how ∇_z T_v is computed:
- **Constant-height (`tz`):** plane-fit T_v interpolated to each z_asl, as in B3. Valid only
  where B3 is.
- **Slope-corrected (`teta`), model levels only:** on each model level k,
  - plane-fit T_v and z_asl over the box to get ∇_η T_v and ∇_η z
  - take ∂T_v/∂z from the tower column (centred difference across k ± 1)
  - ∇_z T_v = ∇_η T_v − (∂T_v/∂z) ∇_η z
  - evaluate at that level's height in the tower column, then interpolate to the output heights
  - Valid down to the lowest model level, about 10 m.

### B5. Large-scale vertical velocity (constant height)

- Box-mean ω at each output height, where valid as in B3.
- w = −ω/(ρ g). The neglected **V**·∇p term is under 1 mm/s.
- **Terrain-relative:** w_terrain = w − **V**·∇z_s.
  - **V**: the box-mean wind at that height.
  - ∇z_s: plane fit of ERA5's orography over the box.
- Hourly values only. Area/time smoothing and picking w at the BL top are left to analysis.

### B6. Profiles at the tower

1. Bilinear interpolation on each model level from the 4 grid points around the tower
   (33.50–33.75 °N, 102.25–102.00 °W). Interpolate heights too, and u and v as components.
2. Interpolate to output heights above ERA5's ground:
   - u, v, T, q: linear in ln z
   - p: ln p linear in z
3. Derived: θ, θ_v (reusing `physics.thermo`).
4. Below the lowest model level (~10 m), nothing is extrapolated; booms 1–3 are left out. The
   surface table carries the 2 m and 10 m diagnostics.

### B7. Surface fields

- Bilinear from the same 4 points.
- **Sign convention:** ERA5 counts vertical fluxes positive downward. `ishf` and `ie` are negated
  to the tower's upward-positive convention.
- **No accumulated fluxes requested.** The instantaneous ones are what's wanted. This also avoids
  the CDS's netCDF conversion splitting instantaneous and accumulated fields into separate
  files.

---

## Part C — Data and configuration

### C1. Downloads

Common to all three:
- **Access:** `cdsapi` with a CDS account. **Elliott accepts the ERA5 licence on the CDS
  website**, then puts the key in `~/.cdsapirc`.
- **Download box:** larger than the stencil, so other stencils can be tried without
  re-downloading: 32.50–34.75 °N, 103.50–100.50 °W.
  - The east extent lets E1 see the escarpment in the orography.
  - Size is negligible for pressure and surface; free for model levels, where cost is per
    global field.
- **Period, in UTC:** 2013-11-01 00:00 to 2014-11-01 23:00, which covers the local-time oneyear
  period.
- **Resumable:** one file per request unit; existing files are skipped; each is written to a
  temporary name and then renamed.

| Command | Dataset | Unit | Contents |
|---|---|---|---|
| `ttu-era5-download surface` | `reanalysis-era5-single-levels` (netCDF) | month | `geopotential` (once; invariant), `forecast_surface_roughness`, `boundary_layer_height`, `instantaneous_surface_sensible_heat_flux`, `instantaneous_moisture_flux`, `friction_velocity`, `instantaneous_eastward/northward_turbulent_surface_stress`, `surface_pressure`, `2m_temperature`, `2m_dewpoint_temperature`, `10m_u/v_component_of_wind`, `100m_u/v_component_of_wind`, `skin_temperature` |
| `ttu-era5-download pressure` | `reanalysis-era5-pressure-levels` (netCDF) | 4 days (cost limit) | z, t, q, u, v, w (ω) at 900, 875, …, 750, 700, …, 400 hPa |
| `ttu-era5-download model` | `reanalysis-era5-complete` (MARS keywords, **GRIB**) | 7 days | params 129/130/131/132/133/135/152; `levelist` = 1 (so `z` and `lnsp` come with it) plus 90–137; `grid` 0.25/0.25; same `area` |

Notes:
- **Model levels:**
  - **Top level (settled at E1):** 90.
    - 5 km above ground is ~478 hPa (2013-11-01), i.e. level ~96 at p_s = 895 hPa.
    - Level 90 is ~376 hPa, about 7 km above ground, which leaves margin.
    - One GRIB day is 9 MB at top level 90.
  - **Request size (settled at E1): 7 days per request**, crossing months (115,248 of the
    120,000 cost limit).
    - In the CDS queue, waiting dominates: one daily job waited 18.5 h and ran 3 min. Daily
      requests (366 waits) would have taken months.
    - The builder skips days already covered by any existing `ml_*` file, so changing the
      chunk size never re-fetches.
    - The GRIB carries the 276-value L137 a/b table (`pv`), so B2 can read it from the data
      instead of shipping a copy.
  - A `parallel` option (2–4 concurrent requests) is worth having for a ~365-request queue.
- **`fsr` varies hourly** (slowly). It's stored hourly; use its mean if a single value is
  wanted.

### C2. Config: `configs/era5.toml`, separate from the tower run config

- **Why separate:** the run-config loader rejects unknown keys, its hashing is tied to tower
  runs, and this tooling is independent of them.

```toml
[paths]
data_dir = "C:/Users/ellwalke/Data/ERA5_TTU"   # surface/, pressure/, model/, processed/

[period]                # UTC
start = "2013-11-01 00:00"
end   = "2014-11-01 23:00"

[download]
area = [34.75, -103.50, 32.50, -100.50]        # N, W, S, E (CDS order)
pressure_levels = [900, 875, 850, 825, 800, 775, 750, 700, 650, 600, 550, 500, 450, 400]
model_level_top = 90
parallel = 3

[process]
source = "model"        # "model" | "pressure", for B3–B5; profiles always need model levels
stencil = { south = 33.25, north = 34.00, west = -102.50, east = -101.75 }
heights_agl = [10.1, 16.8, 47.3, 74.7, 116.5, 158.2, 200.0,
               250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000]
```

Notes:
- **Default heights:** tower booms 4–10, then a ratio-1.25 ladder (rounded) up to 5 km.
- **`source = "pressure"`:** runs B3–B5 without model levels, for E2 while the model-level
  download runs. Profiles are skipped if model levels are missing.

### C3. Outputs, in `<data_dir>/processed/`

**Conventions:** times in UTC, SI units with p in kPa (matching the pipeline), wind components
in true east/north.

- **`era5_profiles.parquet`:** one row per (time, height above ground).
  - **Profile (B6):** `u, v, t, q, p, theta, thetav`, and `w` at the tower.
  - **Geostrophic (B3):** `ug, vg`.
  - **Shear, three estimates:**
    - `dugdz_p, dvgdz_p` (B3)
    - `dugdz_tz, dvgdz_tz` and `dugdz_teta, dvgdz_teta` (B4)
  - **Vertical velocity (B5):** `w_ls, w_ls_terrain`.
  - NaN where a method isn't valid.
- **`era5_surface.parquet`:** one row per time.
  - fluxes, sign-converted
  - `blh, zust, fsr, sp, t2m, d2m, u10, v10, u100, v100, skt`
- **`era5_meta.json`:**
  - the config used, and the source of B3–B5
  - ERA5's ground height at the tower vs 1014 m
  - the stencil, the terrain slope ∇z_s and the plane-fit residual of the terrain
  - the lowest valid height for each method

---

## Part D — Code

```
src/ttu_tower/era5/
  config.py      # era5.toml -> frozen dataclass
  download.py    # three request builders + resumable runner
  levels.py      # B2 heights (a/b from the GRIB pv array)
  columns.py     # pressure/model files -> per-column arrays (B1)
  gradients.py   # plane fit, vg, thermal wind (z and eta), box means
  process.py     # B3-B7, writes C3
src/ttu_tower/cli/era5.py   # ttu-era5-download {surface,pressure,model}; ttu-era5-process
```

- **New optional extra `era5`:** `cdsapi`, `xarray`, `netCDF4`, plus `cfgrib` if model levels
  must come as GRIB.
- Plane fits use numpy least squares; no new numerical dependencies.

**Tests** (targeted, synthetic):
- **Plane fit and B3:** recover a known **v**g from an analytic pressure field over tilted
  terrain.
- **B4:**
  - build terrain-following columns from an analytic atmosphere with a known horizontal T_v
    gradient over a slope
  - check that `teta` and `tz` both recover it, including below the constant-height cutoff for
    `teta`
- **B2 (an integration test on real data):** integrated heights vs pressure-level Φ.

---

## Part E — Phases (Elliott checks after each)

- **E1. Downloads.**
  - Config and the three commands.
  - Run one day of each source, and check:
    - netCDF availability for model levels (GRIB fallback)
    - whether `levelist` with level 1 works
    - variable names in the CDS's netCDF output
    - the model-level top
  - Look at ERA5's orography over the download box to choose the stencil.
  - Then start the full model-level download running.
- **E2. Processing from pressure levels and surface fields:**
  - B1, B3, B4 (`tz`), B5, B7, with `source = "pressure"`.
  - Can be done while the model levels download.
- **E3. Model levels:**
  - B2 and its check against pressure levels
  - B6 profiles
  - B3–B5 from model levels
  - B4 `teta`, and the `tz` vs `teta` comparison above the constant-height cutoff
- **E4 (later, optional). Combined with the tower.** Not designed. Candidates:
  - **Merge ERA5 onto the tower's slots,** like `tertiary/mesonet.py` does for the mesonet.
  - **A tower-informed low-level thermal wind:** ERA5's ∇_η T_v with the **tower's** ∂T_v/∂z
    in the slope term. That replaces ERA5's weakest ingredient (A2.2) with a measurement.
  - **Shear and w at the tower-derived BL height.**

## Open questions

- **Stencil box:** decided at E1 from the orography.
- **Output heights:** whether a ratio-1.25 ladder is dense enough above the tower for
  jet-nose work, or ERA5's native model levels should also be output.
