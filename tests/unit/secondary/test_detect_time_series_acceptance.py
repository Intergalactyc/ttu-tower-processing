"""Statistical check that `detect` behaves on realistic turbulence-plus-waves
time series the way a design-time prototype of the algorithm did: 50
realizations per scenario, comparing the selected-tau flux against the
preset's analytic turbulent covariance and against the same realization's
own turbulence-only flux at the same tau.
"""
import numpy as np
import pytest

from ttu_tower.config.model import DetectionConfig
from ttu_tower.primary import blocks, mrd
from ttu_tower.primary.ladder import RUNG_SECONDS, rung_statistics
from ttu_tower.secondary.detect import detect
from ttu_tower.validation.synthetic import (
    MESOSCALE_STRESS, MESOSCALE_TYPICAL, STABLE_STRESS, STABLE_TYPICAL, turbulence_with_waves,
)

pytestmark = pytest.mark.slow

_SAMPLE_HZ = 50
_N = 3 * 3600 * _SAMPLE_HZ  # 3 h
_WINDOW_SAMPLES = 240_000  # 80 min
_WINDOW_START = (_N - _WINDOW_SAMPLES) // 2  # centered in the 3 h record
_SLOT_START = _WINDOW_START + (_WINDOW_SAMPLES - 30_000) // 2  # the window's central 10-min slot
_FLOOR_LEVEL = 15
_C = 0.75
_FINEST_LEN = 468.75
_FLOOR_LEN = 7.32421875

_DET_CFG = DetectionConfig(peak_significance_se=2.0, min_scale_s=0.5, min_tau_s=9.375, max_tau_s=1200.0)
_SEEDS = range(50)


def _series(ue, vn, w):
    dummy = np.full(ue.size, 290.0)
    return {"ue": ue, "vn": vn, "w": w, "vpts": dummy, "ts": dummy}


def _all_true_masks(n):
    return {"momentum": np.ones(n, bool), "heat": np.ones(n, bool), "ts": np.ones(n, bool)}


def _uw_by_rung(series) -> dict:
    n = series["ue"].size
    b0 = round(_SLOT_START / _FINEST_LEN) - 32
    finest = blocks.finest_sums(series, _all_true_masks(n), g0=0, b0=b0, nb=128)
    values, _ = rung_statistics(finest, _C)
    return {rs: value for rs, variable, stat, value in values if variable == "uw" and stat == "cov"}


def _detect_momentum(series) -> tuple:
    n = series["ue"].size
    b0 = round(_WINDOW_START / _FLOOR_LEN)
    floor = blocks.floor_sums(series, _all_true_masks(n), g0=0, floor_level=_FLOOR_LEVEL, b0=b0, nb=2**_FLOOR_LEVEL)
    spectra, _, _, _ = mrd.detection_outputs(floor, _C)
    D, SE, N = spectra["uw"]
    return detect(D, SE, N, True, _DET_CFG)


def _run_realization(preset, seed) -> dict:
    rng = np.random.default_rng(seed)
    tw = turbulence_with_waves(_N, rng, **preset)
    series_full = _series(8.0 + tw.u, np.zeros(_N), tw.w)

    det = _detect_momentum(series_full)
    result = {"status": det.status, "tau_s": None, "ratio": None, "same_tau": None, "clipped": False,
              "rung600_ratio": None}

    analytic_wu = preset["rho_wu"] * preset["sigma_w"] * preset["sigma_u"]
    full_by_rung = _uw_by_rung(series_full)
    result["rung600_ratio"] = full_by_rung[600.0] / analytic_wu

    if det.status in ("found", "capped"):
        result["tau_s"] = det.tau_s
        full_cov = full_by_rung[det.tau_s]
        series_turb = _series(8.0 + tw.u_turb, np.zeros(_N), tw.w_turb)
        turb_cov = _uw_by_rung(series_turb)[det.tau_s]
        result["ratio"] = full_cov / analytic_wu
        result["same_tau"] = abs(full_cov - turb_cov) <= 0.10 * abs(turb_cov)
        if det.status == "found":
            # "clipped" means the reversal itself, not the pre-clip tau one
            # rung below it, sits at or below min_tau_s: a reversal at mode 7
            # gives the unclipped P_6 = min_tau_s exactly, not a clipped value.
            result["clipped"] = det.reversal_scale_s <= _DET_CFG.min_tau_s + 1e-9

    return result


def _wave_periods(base: dict, extra_period_s: float) -> dict:
    return dict(base, wave_periods_s=(extra_period_s,) + tuple(base["wave_periods_s"]))


_SCENARIOS = {
    "A": dict(preset=MESOSCALE_STRESS, mean_ratio=0.985, found=50, clipped=0, same_tau=50, median_tau=150.0),
    "B": dict(preset=MESOSCALE_TYPICAL, mean_ratio=0.997, found=48, clipped=0, same_tau=50, median_tau=300.0),
    "C": dict(preset=STABLE_STRESS, mean_ratio=0.868, found=50, clipped=0, same_tau=48, median_tau=9.375),
    "D": dict(preset=STABLE_TYPICAL, mean_ratio=0.978, found=48, clipped=0, same_tau=40, median_tau=18.75),
    # E's reversal sits right at the floor for most realizations, exactly
    # where a candidate is most likely to be a noise-level wiggle rather
    # than a real feature (SE can shrink faster than the signal at small
    # scales). Requiring significance there (only there) now lets many
    # realizations find their real, larger-scale reversal instead of
    # flooring on the first insignificant one - clipped and same_tau moved
    # accordingly; mean_ratio and median_tau did not, since the flux at
    # whichever tau ends up selected is largely unchanged.
    "E": dict(preset=_wave_periods(STABLE_STRESS, 30.0), mean_ratio=0.786, found=50, clipped=4, same_tau=14, median_tau=9.375),
    "F": dict(preset=_wave_periods(STABLE_TYPICAL, 30.0), mean_ratio=1.002, found=48, clipped=0, same_tau=34, median_tau=37.5),
    "G": dict(preset=dict(MESOSCALE_STRESS, coupling="aligned"), mean_ratio=0.995, found=50, clipped=0, same_tau=49, median_tau=150.0),
    "H": dict(preset=dict(STABLE_STRESS, coupling="aligned"), mean_ratio=0.955, found=50, clipped=1, same_tau=46, median_tau=9.375),
}


@pytest.mark.parametrize("name", sorted(_SCENARIOS))
def test_time_series_detection_matches_prototype(name):
    spec = _SCENARIOS[name]
    results = [_run_realization(spec["preset"], seed) for seed in _SEEDS]

    active = [r for r in results if r["status"] in ("found", "capped")]
    found_count = sum(r["status"] == "found" for r in results)
    clipped_count = sum(r["clipped"] for r in results)
    same_tau_count = sum(bool(r["same_tau"]) for r in active)
    mean_ratio = float(np.mean([r["ratio"] for r in active])) if active else float("nan")
    median_tau = float(np.median([r["tau_s"] for r in active])) if active else float("nan")
    rung600_mean_ratio = float(np.mean([r["rung600_ratio"] for r in results]))

    report = (
        f"scenario {name}: found={found_count} (target {spec['found']}), "
        f"clipped={clipped_count} (target {spec['clipped']}), "
        f"same_tau={same_tau_count} (target {spec['same_tau']}), "
        f"mean_ratio={mean_ratio:.3f} (target {spec['mean_ratio']}), "
        f"median_tau={median_tau} (target {spec['median_tau']}), "
        f"rung600_mean_ratio={rung600_mean_ratio:.3f} (for information)"
    )

    assert abs(mean_ratio - spec["mean_ratio"]) <= 0.06, report
    assert abs(found_count - spec["found"]) <= 7, report
    assert abs(clipped_count - spec["clipped"]) <= 7, report
    assert abs(same_tau_count - spec["same_tau"]) <= 7, report

    rungs = list(RUNG_SECONDS)
    target_idx = rungs.index(spec["median_tau"])
    allowed = {rungs[i] for i in (target_idx - 1, target_idx, target_idx + 1) if 0 <= i < len(rungs)}
    assert median_tau in allowed, report
