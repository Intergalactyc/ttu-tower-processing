"""Generates tests/fixtures/reference/stage_a_R01457.npz: a Stage A regression
fixture built from the old pipeline's own unit conversion, tilt correction and
rotation, run against real data. Run once with old-tower-processing's own
venv (not the ttu-tower-processing one); never run by the test suite.

    C:\\Users\\ellwalke\\Code\\old-tower-processing\\.venv\\Scripts\\python.exe tests\\fixtures\\make_stage_a_reference.py
"""
import numpy as np
import pandas as pd

from ttu_tower import definitions as d
from ttu_tower.processing.primary_ops import compute_winds, fix_sonic_tilt_misalignment
from windprofiles.processing.units import convert_dataframe_units

RAW_PATH = r"C:\Users\ellwalke\Data\TTU_200m_Nov2013-Oct2014\FT2_E07_C03_R01457_D20131101_T0000.parquet"
OUT_PATH = "tests/fixtures/reference/stage_a_R01457.npz"
N_ROWS = 3000
BOOMS = list(range(1, 11))

VARS = ("u", "v", "w", "ts", "t", "rh", "p")


def main():
    columns = [f"{v}_{b}" for v in VARS for b in BOOMS]
    df = pd.read_parquet(RAW_PATH, columns=columns).iloc[:N_ROWS].reset_index(drop=True)
    df = df.astype(np.float64)  # source is float32; compute in float64 throughout, like the new pipeline

    raw_inputs = {f"{v}_{b}": df[f"{v}_{b}"].to_numpy(dtype=np.float64) for v in VARS for b in BOOMS}

    si = convert_dataframe_units(df, d.SOURCE_UNITS_NEW)
    si = fix_sonic_tilt_misalignment(si, BOOMS)
    out = compute_winds(si, booms_available=BOOMS, old_structure=False)

    outputs = {}
    for b in BOOMS:
        outputs[f"ue_{b}"] = out[f"u_{b}"].to_numpy(dtype=np.float64)
        outputs[f"vn_{b}"] = out[f"v_{b}"].to_numpy(dtype=np.float64)
        outputs[f"w_{b}"] = out[f"w_{b}"].to_numpy(dtype=np.float64)
        outputs[f"ts_{b}"] = si[f"ts_{b}"].to_numpy(dtype=np.float64)
        outputs[f"t_{b}"] = si[f"t_{b}"].to_numpy(dtype=np.float64)
        outputs[f"rh_{b}"] = si[f"rh_{b}"].to_numpy(dtype=np.float64)
        outputs[f"p_{b}"] = si[f"p_{b}"].to_numpy(dtype=np.float64)

    np.savez(OUT_PATH, **raw_inputs, **{f"out_{k}": v for k, v in outputs.items()})
    print(f"wrote {OUT_PATH}: {N_ROWS} rows x {len(BOOMS)} booms")


if __name__ == "__main__":
    main()
