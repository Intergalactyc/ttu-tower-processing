import numpy as np, io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import compare_detect as cd
KAP = 0.2857; PREF = 89.7; HZ = 50
def run(sig_t, rho, trend_kpa_h, seed, noise=0.0):
    rng = np.random.default_rng(seed); n = 3 * 3600 * HZ
    x1, x2 = cd.ou(n, 1.0, rng), cd.ou(n, 1.0, rng)
    w = 0.5 * x1; ts = 290 + sig_t * (rho * x1 + np.sqrt(1 - rho**2) * x2)
    t = np.arange(n) / HZ
    p = 88.0 + trend_kpa_h * (t - t.mean()) / 3600 + noise * rng.standard_normal(n)
    vref = ts * (100 / PREF) ** KAP
    out = []
    for tau, (a, b) in ((600, (n//2 - 15000, n//2 + 15000)), (1200, (n//2 - 30000, n//2 + 30000))):
        L = tau * HZ; sl = slice(a, b)
        def blkcov(x, y):
            xb, yb = x[sl].reshape(-1, L), y[sl].reshape(-1, L)
            return np.mean(((xb - xb.mean(1, keepdims=True)) * (yb - yb.mean(1, keepdims=True))).mean(1))
        f = (PREF / p[sl].mean()) ** KAP
        per_sample = blkcov(w, ts * (100 / p) ** KAP)
        rescaled = f * blkcov(w, vref)
        true = rho * 0.5 * sig_t * (100 / p[sl].mean()) ** KAP
        out.append(((per_sample - rescaled) / abs(true), (rescaled - true) / abs(true)))
    return out
for sig_t, rho, tr in ((0.3, 0.3, 0.5), (0.1, 0.1, 0.5), (0.3, 0.3, 0.1), (0.1, 0.1, 2.0)):
    r = np.array([run(sig_t, rho, tr, s) for s in range(40)])   # (seed, tau, 2)
    for j, tau in enumerate((600, 1200)):
        d = r[:, j, 0]
        print(f"sig_ts={sig_t} rho={rho} dp/dt={tr} kPa/h tau={tau}: (per-sample - rescaled)/|flux|: "
              f"mean {d.mean():+.2e}, rms {np.sqrt((d**2).mean()):.2e}, max {np.abs(d).max():.2e}; "
              f"sampling scatter of the flux itself: rms {np.sqrt((r[:, j, 1]**2).mean()):.2f}")
