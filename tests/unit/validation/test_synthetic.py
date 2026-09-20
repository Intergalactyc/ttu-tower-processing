import numpy as np
import pandas as pd
import pytest

from ttu_tower.constants import SAMPLE_HZ
from ttu_tower.io.load import load_boom
from ttu_tower.primary.stage_a import stage_a
from ttu_tower.validation.synthetic import (
    MESOSCALE_STRESS, MESOSCALE_TYPICAL, STABLE_STRESS, STABLE_TYPICAL, correlated_ou, inject_gaps,
    inject_spikes, inject_stuck, level_shift, ou, quantize, turbulence_with_waves, write_raw_dataset,
    write_raw_files,
)


class _Bounds:
    sonic = (-45.0, 45.0)
    ts = (233.15, 333.15)
    t = (223.15, 323.15)
    rh = (0.0, 1.0)
    p = (50.0, 110.0)


class _QC:
    bounds = _Bounds()


# --- ou / correlated_ou ----------------------------------------------------------

def test_ou_acf_matches_analytic():
    rng = np.random.default_rng(0)
    T = 2.0
    n = 200_000
    x = ou(n, T, sigma=3.0, rng=rng)

    assert x.std() == pytest.approx(3.0, rel=0.03)
    for lag_s in (0.5, 1.0, 2.0, 4.0):
        lag = round(lag_s * SAMPLE_HZ)
        rho = np.corrcoef(x[:-lag], x[lag:])[0, 1]
        assert rho == pytest.approx(np.exp(-lag_s / T), abs=0.05)


def test_correlated_ou_covariance_matches_target():
    rng = np.random.default_rng(1)
    sigmas = [1.0, 2.0, 3.0]
    corr = [[1.0, 0.5, 0.2], [0.5, 1.0, -0.3], [0.2, -0.3, 1.0]]
    x = correlated_ou(200_000, T=1.0, sigmas=sigmas, corr=corr, rng=rng)

    cov = np.cov(x)
    target = np.outer(sigmas, sigmas) * np.array(corr)
    np.testing.assert_allclose(cov, target, rtol=0.1, atol=0.02)


# --- turbulence_with_waves --------------------------------------------------------

def test_turbulence_with_waves_total_variance_additive():
    rng = np.random.default_rng(2)
    n = 12 * 90_000
    tw = turbulence_with_waves(n, rng, **MESOSCALE_TYPICAL)

    turb_var_w = tw.w_turb.var()
    wave_var_w = (tw.w - tw.w_turb).var()
    assert wave_var_w == pytest.approx(MESOSCALE_TYPICAL["wave_std_w"] ** 2, rel=0.05)
    # turbulence and waves are independent draws, so their cross term is only
    # approximately zero in any finite realization.
    assert tw.w.var() == pytest.approx(turb_var_w + wave_var_w, rel=0.01)


@pytest.mark.parametrize("preset", [MESOSCALE_TYPICAL, MESOSCALE_STRESS, STABLE_TYPICAL, STABLE_STRESS])
def test_every_preset_gives_finite_output_with_matching_wave_variance(preset):
    rng = np.random.default_rng(3)
    n = 12 * 90_000
    tw = turbulence_with_waves(n, rng, **preset)

    assert np.isfinite(tw.w).all() and np.isfinite(tw.u).all() and np.isfinite(tw.theta).all()
    wave_var_u = (tw.u - tw.u_turb).var()
    wave_var_w = (tw.w - tw.w_turb).var()
    assert wave_var_u == pytest.approx(preset["wave_std_u"] ** 2, rel=0.05)
    assert wave_var_w == pytest.approx(preset["wave_std_w"] ** 2, rel=0.05)


def test_turbulence_with_waves_opposing_coupling_flux_sign():
    rng = np.random.default_rng(3)
    n = 12 * 90_000
    cfg = dict(MESOSCALE_TYPICAL, coupling="opposing")
    tw = turbulence_with_waves(n, rng, **cfg)

    turb_flux = np.mean(tw.w_turb * tw.u_turb)
    wave_flux = np.mean((tw.w - tw.w_turb) * (tw.u - tw.u_turb))
    assert np.sign(turb_flux) != np.sign(wave_flux)
    assert abs(wave_flux) == pytest.approx(cfg["wave_std_w"] * cfg["wave_std_u"], rel=0.05)


def test_turbulence_with_waves_aligned_coupling_flux_sign():
    rng = np.random.default_rng(4)
    n = 12 * 90_000
    cfg = dict(MESOSCALE_TYPICAL, coupling="aligned")
    tw = turbulence_with_waves(n, rng, **cfg)

    turb_flux = np.mean(tw.w_turb * tw.u_turb)
    wave_flux = np.mean((tw.w - tw.w_turb) * (tw.u - tw.u_turb))
    assert np.sign(turb_flux) == np.sign(wave_flux)
    assert abs(wave_flux) == pytest.approx(cfg["wave_std_w"] * cfg["wave_std_u"], rel=0.05)


def test_turbulence_with_waves_random_coupling_flux_averages_to_zero():
    n = 6 * 90_000
    fluxes = []
    for seed in range(30):
        rng = np.random.default_rng(1000 + seed)
        tw = turbulence_with_waves(n, rng, **MESOSCALE_TYPICAL)
        fluxes.append(np.mean((tw.w - tw.w_turb) * (tw.u - tw.u_turb)))
    assert np.mean(fluxes) == pytest.approx(0.0, abs=0.01)


# --- defect injection --------------------------------------------------------------

def test_inject_stuck_holds_value():
    x = np.arange(10, dtype=np.float64)
    out = inject_stuck(x, start=3, length=4)
    assert out[3:7].tolist() == [3.0, 3.0, 3.0, 3.0]
    assert out[7] == 7.0


def test_inject_gaps_sets_nan():
    x = np.arange(10, dtype=np.float64)
    out = inject_gaps(x, starts=[2, 7], lengths=[2, 1])
    assert np.isnan(out[2:4]).all()
    assert np.isnan(out[7])
    assert not np.isnan(out[4:7]).any()


def test_quantize_rounds_to_step():
    x = np.array([0.0431, 0.0479, -0.021])
    out = quantize(x, 0.01)
    np.testing.assert_allclose(out, [0.04, 0.05, -0.02])


def test_level_shift_adds_from_start():
    x = np.zeros(10)
    out = level_shift(x, start=4, delta=5.0)
    assert out[:4].tolist() == [0.0] * 4
    assert out[4:].tolist() == [5.0] * 6


def test_inject_spikes_alternates_sign_around_median():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1.0, 2000)
    out = inject_spikes(x, positions=[1000], lengths=[4], amplitudes=[10.0])
    diffs = out[1000:1004] - np.median(x)
    assert diffs[0] > 0 and diffs[1] < 0 and diffs[2] > 0 and diffs[3] < 0
    assert abs(diffs[0]) > 5


# --- write_raw_files / write_raw_dataset: round-trip through Stage A -------------

def test_write_raw_dataset_round_trips_through_stage_a(tmp_path):
    rng = np.random.default_rng(6)
    half_hours = [500, 501]
    booms = [1, 4]
    truth = write_raw_dataset(tmp_path, half_hours, booms, rng, waves=MESOSCALE_TYPICAL)

    files = sorted(tmp_path.glob("*.parquet"))
    assert len(files) == 2

    for i, h in enumerate(half_hours):
        for boom in booms:
            raw = load_boom(str(files[i]), boom, temp_dir=str(tmp_path))
            out = stage_a(raw, boom, _QC())
            sl = slice(90_000 * i, 90_000 * (i + 1))
            np.testing.assert_allclose(out.ue, truth[boom]["ue"][sl], atol=1e-4)
            np.testing.assert_allclose(out.vn, truth[boom]["vn"][sl], atol=1e-4)
            np.testing.assert_allclose(out.w, truth[boom]["w"][sl], atol=1e-4)
            np.testing.assert_allclose(out.ts, truth[boom]["ts"][sl], atol=1e-3)
            np.testing.assert_allclose(out.t, truth[boom]["t"][sl], atol=1e-3)
            np.testing.assert_allclose(out.rh, truth[boom]["rh"][sl], atol=1e-4)
            np.testing.assert_allclose(out.p, truth[boom]["p"][sl], atol=1e-3)


def test_write_raw_files_missing_half_hour_writes_no_file(tmp_path):
    rng = np.random.default_rng(7)
    half_hours = [10, 11, 12]
    booms = [1]
    write_raw_dataset(tmp_path, half_hours, booms, rng, missing=[11])
    files = sorted(tmp_path.glob("*.parquet"))
    assert len(files) == 2


def test_write_raw_files_includes_propeller_columns_for_boom_3_plus(tmp_path):
    rng = np.random.default_rng(8)
    write_raw_dataset(tmp_path, [10], [1, 3], rng)
    df = pd.read_parquet(next(tmp_path.glob("*.parquet")))
    assert "propu_3" in df.columns
    assert "propu_1" not in df.columns


def test_mutate_hook_modifies_written_file_not_truth(tmp_path):
    rng = np.random.default_rng(9)

    def mutate(h, boom, df):
        if boom == 2:
            df = df.copy()
            df[f"u_{boom}"] = np.nan
        return df

    truth = write_raw_dataset(tmp_path, [20], [1, 2], rng, mutate=mutate)
    assert np.isfinite(truth[2]["ue"]).all()  # truth unaffected

    df = pd.read_parquet(next(tmp_path.glob("*.parquet")))
    assert df["u_2"].isna().all()
    assert not df["u_1"].isna().any()
