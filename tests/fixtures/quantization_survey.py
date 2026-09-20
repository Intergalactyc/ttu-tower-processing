"""One-off (not run by pytest) quantization survey: for 50 accepted files
spread across the year, per variable, the smallest nonzero step between
distinct Stage A values and the fraction of 5-minute windows whose MAD falls
below the configured min_mad. Checks the min_mad defaults against real data;
the numbers are reported, not asserted.

    .venv\\Scripts\\python.exe tests\\fixtures\\quantization_survey.py
"""
import numpy as np

from ttu_tower.config import load_config
from ttu_tower.constants import BOOMS
from ttu_tower.io.load import load_boom
from ttu_tower.io.rawfiles import accepted_half_hours, build_file_table
from ttu_tower.primary.stage_a import stage_a

CONFIG_PATH = "configs/templates/oneyear.toml"
N_FILES = 50
WINDOW_SAMPLES = 50 * 60 * 5  # 5 minutes
VARS = ("ue", "vn", "w", "ts", "t", "rh", "p")


def _min_nonzero_step(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    diffs = np.diff(np.unique(finite))
    diffs = diffs[diffs > 0]
    return float(diffs.min()) if diffs.size else np.nan


def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def main():
    cfg = load_config(CONFIG_PATH)
    table = build_file_table(cfg.paths.raw_dirs, cfg.files)
    half_hours = accepted_half_hours(table)
    ordered = sorted(half_hours)
    stride = max(1, len(ordered) // N_FILES)
    chosen = ordered[::stride][:N_FILES]

    min_step = {v: np.inf for v in VARS}
    low_mad = {v: 0 for v in VARS}
    total_windows = {v: 0 for v in VARS}

    for h in chosen:
        path = half_hours[h]
        for boom in BOOMS:
            raw = load_boom(path, boom, temp_dir="")
            out = stage_a(raw, boom, cfg.qc)
            for v in VARS:
                x = getattr(out, v)
                step = _min_nonzero_step(x)
                if np.isfinite(step):
                    min_step[v] = min(min_step[v], step)
                n = len(x)
                for start in range(0, n - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
                    window = x[start : start + WINDOW_SAMPLES]
                    if np.all(np.isfinite(window)):
                        total_windows[v] += 1
                        if _mad(window) < cfg.qc.despike.min_mad[v]:
                            low_mad[v] += 1

    print(f"{len(chosen)} files x {len(BOOMS)} booms")
    print(f"{'var':6s} {'min_mad':>10s} {'min_step':>12s} {'n_windows':>10s} {'frac_below_min_mad':>20s}")
    for v in VARS:
        frac = low_mad[v] / total_windows[v] if total_windows[v] else float("nan")
        print(f"{v:6s} {cfg.qc.despike.min_mad[v]:10.6f} {min_step[v]:12.6g} {total_windows[v]:10d} {frac:20.4%}")


if __name__ == "__main__":
    main()
