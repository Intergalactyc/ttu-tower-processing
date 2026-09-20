"""End-to-end check against a direct computation from the generated truth,
covering the four central slots of a clean synthetic run: this catches sign,
unit, frame and offset errors between Stage A, primary and secondary that a
unit test of any one stage in isolation would miss.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from ttu_tower.config import config_from_dict
from ttu_tower.constants import HEIGHTS, SITE_ELEVATION, SITE_LATITUDE
from ttu_tower.io.runs import find_home
from ttu_tower.io.store import read_table
from ttu_tower.math.polar import rotate_streamwise, streamwise_angle, vector_to_bearing
from ttu_tower.physics import most
from ttu_tower.physics.constants import R_CP
from ttu_tower.physics.earth import local_gravity
from ttu_tower.primary.runner import run_primary
from ttu_tower.secondary.runner import run_secondary
from ttu_tower.timegrid import half_hour_to_time
from ttu_tower.validation.synthetic import MESOSCALE_TYPICAL, write_raw_dataset

pytestmark = pytest.mark.slow

_H0 = 7000
_HALF_HOURS = list(range(_H0, _H0 + 6))
_BOOMS = [1, 2]
_CENTRAL_SLOTS = [3 * _H0 + i for i in (7, 8, 9, 10)]  # the run's 18 slots, centered
_SLOT_SAMPLES = 30_000
_REL_TOL = 1e-5
# An exactly-westerly mean wind makes v (and any v-involving covariance)
# genuinely near-zero, dominated by float32/tilt-roundtrip noise at ~1e-12;
# a pure relative tolerance is meaningless there, so every comparison also
# gets this floor.
_ABS_TOL = 1e-8


def _args(config, **overrides):
    base = dict(config=config, test=False, force=False)
    base.update(overrides)
    return SimpleNamespace(**base)


def _cfg(raw_dir, tag):
    start = half_hour_to_time(_HALF_HOURS[0]).strftime("%Y-%m-%d %H:%M")
    end = half_hour_to_time(_HALF_HOURS[-1] + 1).strftime("%Y-%m-%d %H:%M")
    raw = {
        "tag": tag, "paths": {"raw_dirs": [str(raw_dir)]},
        "period": {"start": start, "end": end},
        "files": {"bad_records": [], "booms": _BOOMS},
        "primary": {"batch_max_files": 6},
        # A purely single-component wind (v built only from float32/tilt
        # rounding of an otherwise-exact-zero cross-wind signal) can push a
        # slot's sample kurtosis past the default range by chance - that's a
        # property of this synthetic setup, not a sensor defect, so no
        # quality test should be allowed to remove anything here.
        "qc": {"despike": {"z_threshold": 50.0}, "unusable_tests": []},
        "tertiary": {"veer_reference_boom": 1, "fits": {k: [1] for k in
                     ("alpha_booms", "gamma_booms", "wdgamma_booms", "loglaw_booms")}},
    }
    return config_from_dict(raw)


def _slot_slice(truth_boom: dict, slot: int) -> dict:
    local_lo = _SLOT_SAMPLES * slot - _H0 * 90_000
    local_hi = local_lo + _SLOT_SAMPLES
    return {var: arr[local_lo:local_hi] for var, arr in truth_boom.items()}


def test_secondary_matches_direct_truth_computation(tmp_path, monkeypatch):
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    raw_dir = tmp_path / "raw"
    rng = np.random.default_rng(42)
    truth = write_raw_dataset(raw_dir, _HALF_HOURS, _BOOMS, rng, mean_speed=8.0, mean_from_deg=270.0, waves=MESOSCALE_TYPICAL)

    cfg = _cfg(raw_dir, "sec_truth")
    config_path = tmp_path / "sec_truth.toml"
    config_path.write_text('tag = "sec_truth"\n')
    run_primary(cfg, _args(str(config_path), nproc=1, redo_failures=False, allow_non_parquet=False))
    run_secondary(cfg, _args(str(config_path)))

    run_dir = find_home() / "results" / "sec_truth"
    means = read_table(run_dir / "primary" / "data" / "means")
    boom_stats = read_table(run_dir / "secondary" / "data" / "boom_stats")
    tau_selected = read_table(run_dir / "secondary" / "data" / "tau_selected")
    slow = read_table(run_dir / "secondary" / "data" / "slow")

    # mrd rows exist for every slot (all 18), with a resolved source.
    mrd_selected = tau_selected[tau_selected["variant"] == "mrd"]
    assert set(mrd_selected["slot"]) == set(range(3 * _H0, 3 * _H0 + 18))
    for boom in _BOOMS:
        rows = mrd_selected[mrd_selected["boom"] == boom]
        assert set(rows["slot"]) == set(range(3 * _H0, 3 * _H0 + 18))
        assert rows["source_status"].isin(["found", "capped"]).all(), rows[["slot", "source_status"]]

    def naive_value(slot, boom, variable, stat=None):
        stat_mask = boom_stats["stat"].isna() if stat is None else boom_stats["stat"] == stat
        m = boom_stats[
            (boom_stats["slot"] == slot) & (boom_stats["boom"] == boom) & (boom_stats["variant"] == "naive")
            & (boom_stats["variable"] == variable) & stat_mask
        ]
        assert len(m) == 1, f"expected one ({slot},{boom},{variable},{stat}) naive row, got {len(m)}"
        return float(m["value"].iloc[0])

    def means_value(slot, boom, variable, stat="mean"):
        m = means[(means["slot"] == slot) & (means["boom"] == boom) & (means["variable"] == variable) & (means["stat"] == stat)]
        assert len(m) == 1
        return float(m["value"].iloc[0])

    for slot in _CENTRAL_SLOTS:
        for boom in _BOOMS:
            s = _slot_slice(truth[boom], slot)
            ue, vn, w, ts = s["ue"], s["vn"], s["w"], s["ts"]

            mean_ue, mean_vn, mean_w = ue.mean(), vn.mean(), w.mean()
            phi = streamwise_angle(mean_ue, mean_vn)
            u_p, v_p = rotate_streamwise(ue - mean_ue, vn - mean_vn, phi)
            w_p = w - mean_w

            var_u, var_v, var_w = np.mean(u_p**2), np.mean(v_p**2), np.mean(w_p**2)
            cov_uw, cov_vw, cov_uv = np.mean(u_p * w_p), np.mean(v_p * w_p), np.mean(u_p * v_p)
            ts_var = np.var(ts)

            assert naive_value(slot, boom, "u", "var") == pytest.approx(var_u, rel=_REL_TOL, abs=_ABS_TOL)
            assert naive_value(slot, boom, "v", "var") == pytest.approx(var_v, rel=_REL_TOL, abs=_ABS_TOL)
            assert naive_value(slot, boom, "w", "var") == pytest.approx(var_w, rel=_REL_TOL, abs=_ABS_TOL)
            assert naive_value(slot, boom, "uw", "cov") == pytest.approx(cov_uw, rel=_REL_TOL, abs=_ABS_TOL)
            assert naive_value(slot, boom, "vw", "cov") == pytest.approx(cov_vw, rel=_REL_TOL, abs=_ABS_TOL)
            assert naive_value(slot, boom, "uv", "cov") == pytest.approx(cov_uv, rel=_REL_TOL, abs=_ABS_TOL)
            assert naive_value(slot, boom, "ts", "var") == pytest.approx(ts_var, rel=_REL_TOL, abs=_ABS_TOL)

            p_bar = means_value(slot, boom, "p")
            wvpts_cov_direct = np.mean(w_p * (ts - ts.mean())) * (100.0 / p_bar) ** R_CP
            assert naive_value(slot, boom, "wvpts", "cov") == pytest.approx(wvpts_cov_direct, rel=_REL_TOL, abs=_ABS_TOL)

            ustar_direct = most.friction_velocity(cov_uw, cov_vw)
            assert naive_value(slot, boom, "ustar") == pytest.approx(ustar_direct, rel=_REL_TOL, abs=_ABS_TOL)

            vpt_row = slow[(slow["slot"] == slot) & (slow["boom"] == boom) & (slow["variable"] == "vpt")]
            vpt_slow = float(vpt_row["value"].iloc[0])
            g = local_gravity(SITE_LATITUDE, SITE_ELEVATION + HEIGHTS[boom])
            obukhov_length_direct = most.obukhov_length(ustar_direct, vpt_slow, wvpts_cov_direct, gravity=g)
            assert naive_value(slot, boom, "obukhov_length") == pytest.approx(obukhov_length_direct, rel=_REL_TOL, abs=_ABS_TOL)

            ws_direct = np.mean(np.hypot(ue, vn))
            _, wd_direct = vector_to_bearing(mean_ue, mean_vn)
            assert means_value(slot, boom, "ws") == pytest.approx(ws_direct, rel=_REL_TOL, abs=_ABS_TOL)
            assert means_value(slot, boom, "wd") == pytest.approx(wd_direct, rel=_REL_TOL, abs=_ABS_TOL)
