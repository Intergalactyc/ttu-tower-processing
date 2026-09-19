"""Phase 7 time-series detection scenarios with wave presets grounded in the tower's data
(slow_motion_stats.py and the old one-year 10-min table), run through the exact detection rule.
Per-wave amplitude a = std * sqrt(2 / n_waves), so the waves' total spread equals the observed one.
In phase with a positive w spread, the waves' flux opposes the (negative) turbulent w'u'
("opposing"); a negative w spread makes it the same sign ("aligned")."""
import io, contextlib
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    import compare_detect as cd          # its module-level scenario loop prints; silence it
from detect_spec import detect, P


def waves(std_u, std_w, periods_min):
    k = np.sqrt(2 / len(periods_min))
    return dict(au=std_u * k, aw=std_w * k, periods=periods_min)


MESO_TURB = dict(T=1.0, sw=0.5, su=1.0, rho=-0.35)
STABLE_TURB = dict(T=0.5, sw=0.14, su=0.25, rho=-0.33)
MESO_P, SUB_P, SUB30_P = (20, 30, 45, 60), (1.0, 1.5, 2.5, 4.0), (0.5, 1.0, 1.5, 2.5, 4.0)
MESO_STRESS = {**MESO_TURB, **waves(0.84, 0.067, MESO_P), "shared": True}
MESO_TYPICAL = {**MESO_TURB, **waves(0.56, 0.052, MESO_P), "shared": False}
STABLE_STRESS = {**STABLE_TURB, **waves(0.37, 0.041, SUB_P), "shared": True}
STABLE_TYPICAL = {**STABLE_TURB, **waves(0.23, 0.027, SUB_P), "shared": False}
SCEN = {
    "A  MESOSCALE stress, opposing":      MESO_STRESS,
    "B  MESOSCALE typical, random":       MESO_TYPICAL,
    "C  STABLE stress, opposing":         STABLE_STRESS,
    "D  STABLE typical, random":          STABLE_TYPICAL,
    "E  STABLE stress, opposing, + 30 s": {**STABLE_TURB, **waves(0.37, 0.041, SUB30_P), "shared": True},
    "F  STABLE typical, random, + 30 s":  {**STABLE_TURB, **waves(0.23, 0.027, SUB30_P), "shared": False},
    "G  MESOSCALE stress, aligned":       {**MESO_TURB, **waves(0.84, -0.067, MESO_P), "shared": True},
    "H  STABLE stress, aligned":          {**STABLE_TURB, **waves(0.37, -0.041, SUB_P), "shared": True},
}

if __name__ == "__main__":
    for name, kw in SCEN.items():
        true = kw["rho"] * kw["sw"] * kw["su"]
        st, taus, fs, ft, f600, f1200, clipped = [], [], [], [], [], [], 0
        for s in range(50):
            D, SE, N, u, w, u_t, w_t, slot0 = cd.realization(s, **kw)
            o = detect(D, SE, N)
            tau = o["tau"] if o["status"] in ("found", "capped") else 600.0
            clipped += o["status"] == "found" and o["rev"] <= P[5]
            st.append(f'{o["status"]}/{o["rtype"]}'); taus.append(tau)
            fs.append(cd.flux(u, w, tau, slot0)); ft.append(cd.flux(u_t, w_t, tau, slot0))
            f600.append(cd.flux(u, w, 600.0, slot0)); f1200.append(cd.flux(u, w, 1200.0, slot0))
        fs, ft, f600, f1200, taus = map(np.array, (fs, ft, f600, f1200, taus))
        v, c = np.unique(taus, return_counts=True)
        print(f"{name}: wave flux if in phase {kw['au'] * kw['aw'] / 2 * len(kw['periods']) / true:+.2f} x turbulent; "
              f"{dict((k, st.count(k)) for k in sorted(set(st)))}; tau {dict(zip(v.tolist(), c.tolist()))}; "
              f"median {np.median(taus):g}; clipped {clipped}")
        print(f"   sel/true {np.mean(fs / true):.3f}  600/true {np.mean(f600 / true):.3f}  1200/true {np.mean(f1200 / true):.3f}  "
              f"turb-at-sel/true {np.mean(ft / true):.3f}  |sel/turb-1|<0.1: {int(np.sum(np.abs(fs / ft - 1) < 0.1))}/50")
