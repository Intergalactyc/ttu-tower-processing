"""Amplitude and flux of slow (non-turbulent) motion at the tower, from the old pipeline's one-year
1-min table (read only): per window, the spread of 1-min (or 10-min) means of streamwise u and w,
their covariance (the flux carried at those scales) and its size relative to the turbulent flux
(the window mean of the within-minute w'u'). Grounds the synthetic wave presets."""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

P = r"C:\Users\ellwalke\Code\ttu-tower-processing\results\processed\oneyear_1min\fullyear.parquet"
BOOMS = (2, 4, 6)
cols = ["timestamp", "Rib_2-4"] + [f"{x}_{b}_mean" for b in BOOMS for x in ("u", "wd", "w", "w'u'")]
df = pq.read_table(P, columns=cols).to_pandas()          # timestamp becomes the index
df = df.sort_index()
df = df[~df.index.duplicated()]
grid = pd.date_range(df.index[0].floor("80min"), df.index[-1].ceil("80min"), freq="1min", inclusive="left")
df = df.reindex(grid)


def windows(minutes, avg):
    """Arrays of shape (n_windows, minutes // avg): means over `avg` minutes inside each window."""
    n = len(df) // minutes * minutes
    out = {}
    for c in df.columns:
        a = df[c].to_numpy(float)[:n].reshape(-1, minutes // avg, avg)
        out[c] = a.mean(axis=2)                       # NaN if any minute is missing
    return out


def summarize(label, minutes, avg, classes):
    w = windows(minutes, avg)
    rib = np.nanmedian(w["Rib_2-4"], axis=1)
    print(f"\n### {label}: {minutes}-min windows of {avg}-min means (band ~{2 * avg}-{minutes} min)")
    for cname, (lo, hi) in classes.items():
        m = (rib >= lo) & (rib < hi)
        print(f"  {cname} ({int(m.sum())} windows)")
        for b in BOOMS:
            u, wd, ww, turb = w[f"u_{b}_mean"][m], w[f"wd_{b}_mean"][m], w[f"w_{b}_mean"][m], w[f"w'u'_{b}_mean"][m]
            ok = np.isfinite(u).all(1) & np.isfinite(wd).all(1) & np.isfinite(ww).all(1) & np.isfinite(turb).all(1)
            u, wd, ww, turb = u[ok], wd[ok], ww[ok], turb[ok].mean(1)
            th = np.radians(wd)
            ue, vn = -u * np.sin(th), -u * np.cos(th)
            phi = np.arctan2(vn.mean(1), ue.mean(1))[:, None]
            us = ue * np.cos(phi) + vn * np.sin(phi)
            du, dw = us - us.mean(1, keepdims=True), ww - ww.mean(1, keepdims=True)
            su, sw = du.std(1), dw.std(1)
            cov = (du * dw).mean(1)
            r = cov / (su * sw)
            rel = cov / np.abs(turb)                  # > 0: opposes a downward turbulent flux
            q = lambda x: f"{np.nanquantile(x, .25):+.3g} / {np.nanmedian(x):+.3g} / {np.nanquantile(x, .75):+.3g}"
            print(f"    boom {b} (n={ok.sum()}): std u {q(su)} | std w {q(sw)} | r_uw {q(r)} | "
                  f"slow flux / |turbulent| {q(rel)} | median |slow|/|turb| {np.nanmedian(np.abs(rel)):.2f} | "
                  f"slow flux same sign as turbulent: {np.mean(np.sign(cov) == np.sign(turb)):.2f}")


summarize("Submesoscale", 20, 1, {"stable": (0.01, 0.1), "strongly stable": (0.1, np.inf)})
summarize("Mesoscale", 80, 10, {"unstable": (-np.inf, -0.01), "neutral": (-0.01, 0.01)})
