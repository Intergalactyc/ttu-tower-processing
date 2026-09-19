# Updated TTU 200 m tower processing pipeline
### Top priority goals:
- QC redone
    - Modified procedure
    - Filter flags at boom level!!!
- MRD-based variance and covariance estimation (for proper fluxes and TI)
- Removal of windprofiles dependency (necessary code brought over)
    - As a result, no more need to adapt to its exact formatting (can avoid need for measurements.legacy_adapter and such)
### Lower priority goals:
- Cage shadowing
- PSD calculations and spectral corrections (aliasing, path length attenuation [Moore])
- Proper GUI for result investigation?
### Additional differences:
- Use response smoothing for slow variables
- Ignore propeller anemometers altogether?
- Move bad record removal to initial loading step (don't bother processing them in the first place)
- Tertiary applies quality flags at boom level (needed for profile fits); stability classification is left for post-pipeline steps
    - Anisotropy calculations move from tertiary to secondary
### Ideals
- Cleaner codebase
    - Better organization
    - Readability prioritized
    - No bloat, no clutter (including comment clutter)
- Validation
    - Code validation: proper test suite
    - QC validation
### Further ideas
- Monte Carlo framework for synthetic data, and modification thereof for QC validation?

## QC steps
### Corrective QC:
- Tilt correction
- t, rh, p smoothing over response timescales
- MAYBE: angle of attack correction (Nakai & Shimoyama, 2012)
- Bounds enforcement (absolute limits)
- Despiking (Mauder & Foken, 2015 [TK3 white paper]: sec 3.1.2?)
- Treatment of missing values (interior interpolation, exterior truncation) (up to certain total % missing, then flag - MF15 sec 3.2.2)

### Primary QC flags:
- Amplitude resolution
- Dropouts
- Higher moments
- MAYBE: discontinuities (using Haar transform) (Vickers & Mahrt, 1996: section 6f)

### Secondary QC flags:
- Tower shadowing
- MAYBE: cage shadowing
- Signal bouncing

## MRD procedure (distinct per boom!!!):
- Apply corrective QC and compute raw QC flags
- Interpolate onto next largest 2^M grid size (for 50 Hz, 30 min - use 2^17 = 131072, this is then 0.01373 s/measurement) (as in Howell & Mahrt, 1995 [MRD paper]: sec 4.1)
- Compute heat flux cospectra using MRD
- Use heat flux cospectra to determine gap timescale tau (corresponding to 2^m grid size for some 1 <= m < M; possibly restrict to m > 8 [2^8 on new grid corresponds to ~3.5 s] to ensure timescale always at least 5 s)
    - If no clear gap timescale identifiable, flag record
- Within intervals of length tau, perform mean-wind alignment on u and v
    - After this, compute momentum flux cospectra (and gap timescale, to see if it differs at all??) too?
- Compute covariances (and variances) using timescale tau
    - Heat & momentum fluxes (w'vpts', w'u', w'v', u'v')
    - Covariances --> rms values (u'u', v'v', w'w')
        - Also wd rms
    - Transport efficiencies
- Other calculations using timescale tau
    - Autocorrelations --> integral timescales
        - Average of integral timescales calculated over each sub-interval?

## Order of operations:
### Primary
- Data loading, header map, unit conversion
- Corrective QC
- Primary QC flags
- High-frequency variable calculations
    - vpts
    - wd
- MRD procedure to determine gap timescale
    - Save cospectra to disk?
- Covariances, variances, transport efficiencies
- Test sub-interval stationarity
- Calculate autocorrelations->integral timescales over sub-intervals and aggregate
    - Restrict aggregation to stationary sub-intervals?
    - u, v, w, and ts timescales
- Calculate PSD
    - Save binned PSD to disk
- Summarization
    - Means of all things
    - (RMSs, covariances, TEs, ITSs already from above)
    - Naive RMSs (and covariances?)
    - Wind speed maxes for gust factor?
- Other flags?
    - Steady state / ITC (/--> Spoleto)??
        - ITC downside: need to then compute L and ustar here
        - Now that MRD used for timescales, and stationarity handled elsewhere, maybe no more need for this?

### Secondary
- Secondary QC flags
    - Shadowing flag(s)
    - Signal bouncing flag
- Single-boom derived quantities
    - VPT (using slow sensor results)
        - PT and VT as well!!
        - Other moisture quantities? (Td, q, w, e, e_s, ...)
    - Integral length scales
    - Stability quantities: L, z/L (sparam)
- Boom-independent quantities
    - Sun elevation
- Anisotropy calculations

### Tertiary
- Mesonet merge
- By-boom flag filtering step
- Profile fits
- Any mesonet-derived quantities
- Other multi-boom derived quantities
    - Shear and veer measures
    - Lapse rate measures
    - Ri_b

### Post (have helper structures/functions, but not part of pipeline itself)
- Stability classification
- Summarization (as is currently in ttu-windprofiles)?
- Visualization??
- Annual/summary profile fits; general curve fits??
- ... Actually, maybe the summarization, visualization, and summary profile/curve fits (and further such tools which could be used between different post-pipelne analyses) could be scoped in a new repo? (But do provide at least stability classification here)

## Where the code will come from
### What can be reused with minor modification from original repos
ttu-tower-processing:
- Config system
- Primary-secondary-tertiary split (the general idea)
    - Mesonet merge logic
- Most of the IO system
- Raw->parquet conversion code
- CLI entry points
windprofiles:
- Multiprocessing and logging frameworks
- Stability classifier system

### What can be reused with notable modification from original repos
- Interactive mode
    - Don't worry about this yet
        - Low priority; just need to keep in mind so that it can later be easily built upon what exists (e.g. loading step compatibility)
- QC report system
- Some QC test implementations
- Some code tests
- Flag and FlagFilter system
    - Flag filtering does need to be fairly different do work at boom level
        - Also, different stages of filtering needed...
    - Lots of questions here
windprofiles:
- Autocorrelation and integral time scales
    - Maybe trim down this system (not so many methods? or keep most/all, but always do compare, thus simplifying config and logic?)
- Various derived quantity calculations / summarization steps

### What needs to be done largely from scratch
- MRD system
- PSD system
- Other QC test implementations
- Further code tests

### What can be left out from original repos
- Existing ttu-tower-processing prototyping system
- Propeller data processing
    - The columns still exist - just don't actually process them or otherwise use them for anything
    - Still should exist in parquets created in raw->parquet conversion step
    - But on dataset loading for actual processing, drop their columns as they're unneeded

## Questions that remain
- Where to pen in max wind speed calculations for things like gust factor determination? (Relates to below 10/60 min standardization problem)
- 10/60-min "standardization" problem?
    - Typical reports made on 10 min or 60 min intervals
    - Option 1 - chunk 30 to 2x10 minutes before anything else [use M=15]
        - Issue: gap timescales greater than 10 minutes (possible in highly convective conditions) would then result in underestimation of turbulent (co)variances
    - Option 2 - join 2x30 to 60 minutes before anything else [use M=18]
        - Issue: 10 min is a bit more typical?
        - Issue: more time for conditions to vary (stability may change entirely over an hour)
    - Option 3 - have two separate things - above procedure (at 10, 30, or 60 min - basic way or use one of Options 1-2) and naive procedure at 10 min (or both 10 and 60 min)
        - Decent bit more work
        - Not much reason to do this rather than just using Option 1 or 2?
    - Any other solutions?
    - For opts 1 or 2, also compute "naive" RMS values for comparability; max wind speeds for GF and such can just be maxes in this interval
        - Maybe also compute naive covariances?
- How exactly to handle stationarity for autocorrelation calculations?
    - Further, stationarity for full time interval flags?
- Nonrobustness of mean itself over nonstationary 30 (/10/60) min interval?
    - Related and more consequential: stability may vary over this time interval
- A lot of thought needs to be given to how the flag filtering system should work
    - Integration with profile fit step
        - For example: for wind speed profile fits, we don't need to exclude a boom for failing stationarity tests or having a broken pressure sensor, while for TI profiles stationarity may be necessary, and for length scales it absolutely is (but also, the time scales may only have been computed using the stationary sub-intervals? so this ties in to an above question)...
    - Aaah... also in e.g. user-performed summarization (e.g. for creating annual profiles, perhaps with stability dependence too), certain booms need to be excluded based on their flags
        - Oh - all that needs to be done is replace any affected values with NaNs (at filter time in tertiary step)!
            - Then filter-unaware summarization, as long as it is NaN-safe, will ignore bad values just fine
            - Challenge: determine the list of affected values corresponding to each type of issue (e.g. bad quality -> nothing works, but instationary -> only 2nd-order and above bad)
                - Maybe not that hard. Actually may just be 2 types of removals to perform: quality-based and stationarity-based
            - And this cleans up the profile fit step too: don't need to integrate flag filtering with it, just perform it before, and profile fits just have to ensure there are sufficient non-NaN booms for their given variable
    - Configurability (for user) before running tertiary (with a config section or a whole new config?)
    - Worth also providing a "general" filtering system (for use post-pipeline, not in pipeline) that does exclude entire records (e.g. if >N booms have quality issues, or quality issues of a certain kind are present?)
- Probably plenty of other things I haven't thought of
