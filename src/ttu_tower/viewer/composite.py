"""Composite MRD spectra: each boom's spectra over many slots, summarized per
scale as a median with interquartile bars (as ttu-windprofiles' cospectra
figure does), over a range of slots, a stability class or a brushed selection.
Transport efficiencies per ladder rung composite the same way.
"""
import warnings

import numpy as np
import pandas as pd
import pyarrow.dataset as pa_dataset

SPECTRA = (("uu", "u variance"), ("vv", "v variance"), ("ww", "w variance"), ("tsts", "Ts variance"),
           ("uw", "u'w' cospectrum"), ("vw", "v'w' cospectrum"), ("uv", "u'v' cospectrum"),
           ("wvpts", "w'θv' cospectrum"))
VARIANCES = ("uu", "vv", "ww", "tsts")
TRANSPORT = (("uw_te", "u'w' transport efficiency"), ("wvpts_te", "w'θv' transport efficiency"))
SCALES = (("its_u", "ITS u", "s"), ("its_v", "ITS v", "s"), ("its_w", "ITS w", "s"), ("its_vpts", "ITS θv", "s"),
          ("ils_u", "ILS u", "m"), ("ils_v", "ILS v", "m"), ("ils_w", "ILS w", "m"), ("ils_vpts", "ILS θv", "m"))
REFERENCE_SCALES = (60.0, 180.0, 600.0, 1800.0)  # the paper's dashed guides [s]
LABELS = dict(SPECTRA + TRANSPORT)
UNITS = {"uu": "m²/s²", "vv": "m²/s²", "ww": "m²/s²", "uw": "m²/s²", "vw": "m²/s²", "uv": "m²/s²", "tsts": "K²",
         "wvpts": "K m/s"}  # per mode; transport efficiencies have none
TE_TIP = ("transport efficiency at each ladder rung (fluctuations about that rung's block means): the net flux over "
          "the flux of its dominant sign, 1 when every sample's product has the same sign")


def spectra_matrix(run, index, boom: int, variant: str, spectrum: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(slots, scales, values[slot, scale]) of one boom's stored spectrum over the
    whole period (NaN where a slot lacks a scale). Cached on the run.
    """
    key = ("spectra", boom, variant, spectrum)
    return run.cache("composite", 48).get(key, lambda: _matrix(index, boom, variant, spectrum))


def _matrix(index, boom, variant, spectrum):
    f = pa_dataset.field
    if spectrum in dict(TRANSPORT):  # per ladder rung, from primary's ladder table
        df = index.read("ladder", booms=[boom], columns=["slot", "rung_s", "value"],
                        filter=(f("variant") == variant) & (f("variable") == spectrum[:-len("_te")])
                        & (f("stat") == "te")).rename(columns={"rung_s": "scale_s"})
    else:
        df = index.read("mrd", booms=[boom], columns=["slot", "scale_s", "value"],
                        filter=(f("variant") == variant) & (f("spectrum") == spectrum))
    if df.empty:
        return np.empty(0, dtype=np.int64), np.empty(0), np.empty((0, 0))
    slots, row = np.unique(df["slot"].to_numpy(dtype=np.int64), return_inverse=True)
    scales, col = np.unique(df["scale_s"].to_numpy(dtype=float), return_inverse=True)
    values = np.full((slots.size, scales.size), np.nan)
    values[row, col] = df["value"].to_numpy(dtype=float)
    return slots, scales, values


def scales_matrices(run, index, boom: int, variant: str) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """(slots, rungs, {quantity: values[slot, rung]}) of one boom's integral
    scales at every ladder rung over the whole period: the ITS of u, v, w and
    θv, and ILS = ITS × |u mean at that rung| (as secondary defines it).
    Cached on the run.
    """
    key = ("scales", boom, variant)
    return run.cache("composite", 48).get(key, lambda: _scales(index, boom, variant))


def _scales(index, boom, variant):
    f = pa_dataset.field
    rows = index.read("ladder", booms=[boom], columns=["slot", "rung_s", "variable", "stat", "value"],
                      filter=(f("variant") == variant) & (f("boom") == boom)
                      & (((f("stat") == "its") & f("variable").isin(["u", "v", "w", "vpts"]))
                         | ((f("stat") == "mean") & (f("variable") == "u"))))
    if rows.empty:
        return np.empty(0, dtype=np.int64), np.empty(0), {}
    slots, row = np.unique(rows["slot"].to_numpy(dtype=np.int64), return_inverse=True)
    rungs, col = np.unique(rows["rung_s"].to_numpy(dtype=float), return_inverse=True)
    names = (rows["stat"].astype(str) + ":" + rows["variable"].astype(str)).to_numpy()
    values = rows["value"].to_numpy(dtype=float)
    out = {}
    for name in ("its:u", "its:v", "its:w", "its:vpts", "mean:u"):
        m = np.full((slots.size, rungs.size), np.nan)
        sel = names == name
        m[row[sel], col[sel]] = values[sel]
        out[name] = m
    speed = np.abs(out.pop("mean:u"))
    for var in ("u", "v", "w", "vpts"):
        out[f"its_{var}"] = out.pop(f"its:{var}")
        out[f"ils_{var}"] = out[f"its_{var}"] * speed
    return slots, rungs, out


def summarize(slots, scales, values, keep: np.ndarray, normalize: bool = False,
              spread: tuple[float, float] = (25.0, 75.0), center: str = "median") -> pd.DataFrame:
    """Per scale: median, the `spread` percentiles and the number of slots, over
    the rows `keep` selects; or, with `center="mean"`, the mean ± one standard
    deviation. `normalize` divides each slot's spectrum by its sum over scales
    (the within-block (co)variance), so slots of different intensity weigh alike.
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
        if center == "mean":
            mid, sd = np.nanmean(rows, axis=0), np.nanstd(rows, axis=0)
            out["mid"], out["lo"], out["hi"] = mid, mid - sd, mid + sd
        else:
            out["mid"] = np.nanmedian(rows, axis=0)
            out["lo"], out["hi"] = np.nanpercentile(rows, spread, axis=0)
    return out
