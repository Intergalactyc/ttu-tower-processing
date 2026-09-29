"""Composite MRD spectra: each boom's spectra over many slots, summarized per
scale as a median with interquartile bars (as ttu-windprofiles' cospectra
figure does), over a range of slots, a stability class or a brushed selection.
"""
import warnings

import numpy as np
import pandas as pd
import pyarrow.dataset as pa_dataset

SPECTRA = (("uu", "u variance"), ("vv", "v variance"), ("ww", "w variance"), ("tsts", "Ts variance"),
           ("uw", "u'w' cospectrum"), ("vw", "v'w' cospectrum"), ("uv", "u'v' cospectrum"),
           ("wvpts", "w'θv' cospectrum"))
VARIANCES = ("uu", "vv", "ww", "tsts")
REFERENCE_SCALES = (60.0, 180.0, 600.0, 1800.0)  # the paper's dashed guides [s]


def spectra_matrix(run, index, boom: int, variant: str, spectrum: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(slots, scales, values[slot, scale]) of one boom's stored spectrum over the
    whole period (NaN where a slot lacks a scale). Cached on the run.
    """
    key = ("spectra", boom, variant, spectrum)
    return run.cache("composite", 48).get(key, lambda: _matrix(index, boom, variant, spectrum))


def _matrix(index, boom, variant, spectrum):
    f = pa_dataset.field
    df = index.read("mrd", booms=[boom], columns=["slot", "scale_s", "value"],
                    filter=(f("variant") == variant) & (f("spectrum") == spectrum))
    if df.empty:
        return np.empty(0, dtype=np.int64), np.empty(0), np.empty((0, 0))
    slots, row = np.unique(df["slot"].to_numpy(dtype=np.int64), return_inverse=True)
    scales, col = np.unique(df["scale_s"].to_numpy(dtype=float), return_inverse=True)
    values = np.full((slots.size, scales.size), np.nan)
    values[row, col] = df["value"].to_numpy(dtype=float)
    return slots, scales, values


def summarize(slots, scales, values, keep: np.ndarray, normalize: bool = False,
              spread: tuple[float, float] = (25.0, 75.0)) -> pd.DataFrame:
    """Per scale: median, the `spread` percentiles and the number of slots, over
    the rows `keep` selects. `normalize` divides each slot's spectrum by its sum
    over scales (the within-block (co)variance), so slots of different intensity
    weigh alike.
    """
    rows = values[keep]
    if normalize and rows.size:
        with np.errstate(invalid="ignore", divide="ignore"):
            total = np.nansum(rows, axis=1, keepdims=True)
            rows = np.where(np.abs(total) > 0, rows / total, np.nan)
    n = np.isfinite(rows).sum(axis=0) if rows.size else np.zeros(scales.size, dtype=int)
    out = pd.DataFrame({"scale_s": scales, "n": n})
    if rows.shape[0] == 0:
        out["mid"] = out["lo"] = out["hi"] = np.nan
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out["mid"] = np.nanmedian(rows, axis=0)
        out["lo"], out["hi"] = np.nanpercentile(rows, spread, axis=0)
    return out
