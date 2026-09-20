"""Synthetic data generators for tests and first-run validation: known-answer
turbulence, defect injection, and synthetic raw files the pipeline reads like
real ones.
"""
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from ttu_tower import __version__
from ttu_tower.constants import SAMPLE_HZ
from ttu_tower.io.store import write_json_atomic
from ttu_tower.math.polar import bearing_to_vector, rotate_streamwise, streamwise_angle
from ttu_tower.primary.stage_a import INHG_TO_KPA, MPH_TO_MS, tilt_matrix
from ttu_tower.timegrid import SAMPLES_PER_HALF_HOUR, half_hour_to_time

# --- Ornstein-Uhlenbeck processes -----------------------------------------------


def ou(n: int, T: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """A stationary OU process (ACF = exp(-lag/T)): the exact AR(1)
    discretization x_{k+1} = a*x_k + sigma*sqrt(1-a^2)*eps_k, a = exp(-dt/T),
    x_0 drawn from the stationary distribution.
    """
    dt = 1.0 / SAMPLE_HZ
    a = np.exp(-dt / T)
    b = sigma * np.sqrt(1.0 - a**2)
    x = np.empty(n, dtype=np.float64)
    x[0] = rng.normal(0.0, sigma)
    if n > 1:
        eps = rng.normal(0.0, 1.0, n - 1)
        driven = lfilter([b], [1.0, -a], eps)
        x[1:] = a ** np.arange(1, n) * x[0] + driven
    return x


def correlated_ou(n: int, T: float, sigmas, corr, rng: np.random.Generator) -> np.ndarray:
    """`len(sigmas)` OU processes sharing timescale T, correlated through the
    Cholesky factor of `corr`: covariance sigma_i*sigma_j*corr_ij, each with
    ACF exp(-lag/T). Returns shape (len(sigmas), n).
    """
    sigmas = np.asarray(sigmas, dtype=np.float64)
    m = sigmas.size
    L = np.linalg.cholesky(np.asarray(corr, dtype=np.float64))
    unit = np.stack([ou(n, T, 1.0, rng) for _ in range(m)], axis=0)
    return (L @ unit) * sigmas[:, None]


# --- Turbulence plus waves -------------------------------------------------------


@dataclass
class TurbulenceWaves:
    w: np.ndarray
    u: np.ndarray
    theta: np.ndarray
    w_turb: np.ndarray
    u_turb: np.ndarray
    theta_turb: np.ndarray


MESOSCALE_TYPICAL = dict(
    T=1.0, sigma_w=0.5, sigma_u=1.0, sigma_theta=0.3, rho_wu=-0.35, rho_wtheta=0.3,
    wave_periods_s=(1200.0, 1800.0, 2700.0, 3600.0),
    wave_std_u=0.56, wave_std_w=0.052, wave_std_theta=0.28, coupling="random",
)


def turbulence_with_waves(n: int, rng: np.random.Generator, *, T: float, sigma_w: float, sigma_u: float,
                           sigma_theta: float, rho_wu: float, rho_wtheta: float, wave_periods_s,
                           wave_std_u: float, wave_std_w: float, wave_std_theta: float, coupling: str) -> TurbulenceWaves:
    """Turbulent (w, u, theta) from `correlated_ou`, plus non-turbulent
    "waves": one sinusoid per period per variable, random phase and amplitude
    std*sqrt(2/p) (p periods), so their total spread equals the given std.
    `coupling` sets the (w, u) phase difference: "random" (independent
    phases), "opposing" (wave flux opposes the turbulent w'u' sign, carrying
    the full flux std_w*std_u), or "aligned" (matches it). theta's phase is
    always independent of w, since only the momentum cospectrum is used for
    detection.
    """
    corr = np.array([[1.0, rho_wu, rho_wtheta], [rho_wu, 1.0, 0.0], [rho_wtheta, 0.0, 1.0]])
    turb = correlated_ou(n, T, [sigma_w, sigma_u, sigma_theta], corr, rng)
    w_turb, u_turb, theta_turb = turb[0], turb[1], turb[2]

    t = np.arange(n) / SAMPLE_HZ
    p = len(wave_periods_s)
    amp_u = wave_std_u * np.sqrt(2.0 / p)
    amp_w = wave_std_w * np.sqrt(2.0 / p)
    amp_theta = wave_std_theta * np.sqrt(2.0 / p)

    if coupling == "random":
        delta_phi = None
    else:
        turb_sign = 1.0 if rho_wu >= 0 else -1.0
        desired_sign = -turb_sign if coupling == "opposing" else turb_sign
        delta_phi = 0.0 if desired_sign >= 0 else np.pi

    w_wave = np.zeros(n)
    u_wave = np.zeros(n)
    theta_wave = np.zeros(n)
    for period_s in wave_periods_s:
        omega = 2.0 * np.pi / period_s
        phi_w = rng.uniform(0.0, 2.0 * np.pi)
        phi_theta = rng.uniform(0.0, 2.0 * np.pi)
        phi_u = rng.uniform(0.0, 2.0 * np.pi) if delta_phi is None else phi_w + delta_phi
        w_wave += amp_w * np.sin(omega * t + phi_w)
        u_wave += amp_u * np.sin(omega * t + phi_u)
        theta_wave += amp_theta * np.sin(omega * t + phi_theta)

    return TurbulenceWaves(
        w=w_turb + w_wave, u=u_turb + u_wave, theta=theta_turb + theta_wave,
        w_turb=w_turb, u_turb=u_turb, theta_turb=theta_turb,
    )


# --- Defect injection -------------------------------------------------------------


def inject_spikes(x: np.ndarray, positions, lengths, amplitudes) -> np.ndarray:
    """Runs of `lengths[i]` samples (1-6 typical) at `positions[i]`, offset
    from the local median by `amplitudes[i]` times the local robust sigma
    (MAD/0.6745 over a +-5 s window), alternating sides of the median sample
    by sample.
    """
    x = np.array(x, dtype=np.float64, copy=True)
    half_window = 5 * SAMPLE_HZ
    for pos, length, amp in zip(positions, lengths, amplitudes):
        lo, hi = max(0, pos - half_window), min(x.size, pos + half_window)
        window = x[lo:hi]
        median = np.nanmedian(window)
        sigma = np.nanmedian(np.abs(window - median)) / 0.6745
        for i in range(length):
            sign = 1.0 if i % 2 == 0 else -1.0
            x[pos + i] = median + sign * amp * sigma
    return x


def inject_stuck(x: np.ndarray, start: int, length: int) -> np.ndarray:
    """Holds x[start] constant for `length` samples (a dropout run)."""
    x = np.array(x, dtype=np.float64, copy=True)
    x[start : start + length] = x[start]
    return x


def inject_gaps(x: np.ndarray, starts, lengths) -> np.ndarray:
    """Sets [start, start+length) to NaN for each (start, length)."""
    x = np.array(x, dtype=np.float64, copy=True)
    for s, length in zip(starts, lengths):
        x[s : s + length] = np.nan
    return x


def quantize(x: np.ndarray, step: float) -> np.ndarray:
    """Rounds x to the nearest multiple of `step`."""
    return np.round(np.asarray(x, dtype=np.float64) / step) * step


def level_shift(x: np.ndarray, start: int, delta: float) -> np.ndarray:
    """Adds `delta` to every sample from `start` onward (a lasting excursion)."""
    x = np.array(x, dtype=np.float64, copy=True)
    x[start:] += delta
    return x


# --- Synthetic raw files -----------------------------------------------------------

_PROPELLER_VARS = ("propu", "propv", "propw")


def _to_native(boom: int, ue, vn, w, ts, t, rh, p) -> dict[str, np.ndarray]:
    """Inverts Stage A for one boom: rotation, tilt (transpose), then source units."""
    u_native, v_native = -vn, ue
    r = tilt_matrix(boom)
    u0 = r[0, 0] * u_native + r[1, 0] * v_native + r[2, 0] * w
    v0 = r[0, 1] * u_native + r[1, 1] * v_native + r[2, 1] * w
    w0 = r[0, 2] * u_native + r[1, 2] * v_native + r[2, 2] * w
    return {
        "u": u0 / MPH_TO_MS,
        "v": v0 / MPH_TO_MS,
        "w": w0 / MPH_TO_MS,
        "ts": (ts - 273.15) * (9.0 / 5.0) + 32.0,
        "t": (t - 273.15) * (9.0 / 5.0) + 32.0,
        "rh": rh * 100.0,
        "p": p / INHG_TO_KPA,
    }


def write_raw_files(dir, series: dict[int, dict[str, np.ndarray]], half_hours, booms,
                     missing=(), mutate=None) -> None:
    """Writes `series` (per-boom SI dicts, each covering the contiguous span
    from min(half_hours) to max(half_hours)) as converted-style Parquet raw
    files, named like the real convention with increasing record numbers.
    `missing` half-hours get no file. `mutate(half_hour, boom, df)` may
    modify a boom's native-unit columns before they're written.
    """
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    missing = set(missing)
    sorted_hh = sorted(half_hours)
    h0 = sorted_hh[0]

    record = 1
    for h in sorted_hh:
        if h in missing:
            continue
        boom_frames = []
        for boom in booms:
            s = series[boom]
            sl = slice((h - h0) * SAMPLES_PER_HALF_HOUR, (h - h0 + 1) * SAMPLES_PER_HALF_HOUR)
            native = _to_native(boom, s["ue"][sl], s["vn"][sl], s["w"][sl], s["ts"][sl], s["t"][sl], s["rh"][sl], s["p"][sl])

            boom_df = pd.DataFrame({f"{var}_{boom}": native[var].astype(np.float32) for var in native})
            if boom >= 3:
                for pv in _PROPELLER_VARS:
                    boom_df[f"{pv}_{boom}"] = np.float32(np.nan)

            if mutate is not None:
                boom_df = mutate(h, boom, boom_df)
            boom_frames.append(boom_df)

        file_df = pd.concat(boom_frames, axis=1)
        file_start = half_hour_to_time(h)
        name = f"FT2_E07_C03_R{record:05d}_D{file_start.strftime('%Y%m%d')}_T{file_start.strftime('%H%M')}.parquet"
        file_df.to_parquet(dir / name)
        record += 1


def write_raw_dataset(dir, half_hours, booms, rng: np.random.Generator, *, mean_speed: float = 8.0,
                       mean_from_deg: float = 270.0, waves=None, missing=(), mutate=None) -> dict[int, dict[str, np.ndarray]]:
    """Generates a dataset the pipeline reads like a real one: each boom's SI
    series (Earth-frame ue, vn, w; ts; slow t/rh/p), written as raw files by
    `write_raw_files`. Returns `truth`: the generated SI series per boom,
    before inversion.
    """
    if waves is None:
        waves = MESOSCALE_TYPICAL
    half_hours = list(half_hours)
    h0, h1 = min(half_hours), max(half_hours)
    n = (h1 - h0 + 1) * SAMPLES_PER_HALF_HOUR

    mean_ue, mean_vn = bearing_to_vector(mean_speed, mean_from_deg)
    phi = streamwise_angle(mean_ue, mean_vn)

    truth = {}
    for boom in booms:
        tw = turbulence_with_waves(n, rng, **waves)
        u_earth, v_earth = rotate_streamwise(tw.u, np.zeros(n), -phi)
        ue = mean_ue + u_earth
        vn = mean_vn + v_earth
        w = tw.w
        ts = 290.0 + tw.theta

        t = quantize(np.clip(285.0 + np.cumsum(rng.normal(0.0, 2e-5, n)), 250.0, 320.0), 0.002)
        rh = quantize(np.clip(0.5 + np.cumsum(rng.normal(0.0, 2e-7, n)), 0.02, 0.98), 0.00002)
        p = quantize(np.clip(90.0 + np.cumsum(rng.normal(0.0, 5e-6, n)), 55.0, 105.0), 0.0004)

        truth[boom] = {"ue": ue, "vn": vn, "w": w, "ts": ts, "t": t, "rh": rh, "p": p}

    write_raw_files(dir, truth, half_hours, booms, missing=missing, mutate=mutate)
    write_json_atomic(Path(dir) / "synthetic.json", {
        "generator": "write_raw_dataset",
        "half_hours": [h0, h1],
        "booms": list(booms),
        "mean_speed": mean_speed,
        "mean_from_deg": mean_from_deg,
        "waves": {k: (list(v) if isinstance(v, tuple) else v) for k, v in waves.items()},
        "missing": sorted(missing),
        "numpy_version": np.__version__,
        "package_version": __version__,
        "python_version": platform.python_version(),
    })
    return truth
