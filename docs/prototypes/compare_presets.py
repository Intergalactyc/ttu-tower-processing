"""The detection-rule comparison of design-reference H, on the data-grounded presets of
presets_spec.py (50 seeds). Rules: the earlier significance tests on peak and reversal
(compare_detect.plan_detect), pure VM06 (detect_spec with the peak test off, k = 0), and the
adopted VM06 + peak test (detect_spec as specified). Momentum cospectrum; a `weak` or
`unresolved` result falls back to 600 s, as selection would with one source."""
import numpy as np
from presets_spec import cd, detect, MESO_STRESS, MESO_TYPICAL, STABLE_STRESS, STABLE_TYPICAL


def adopted(D, SE, N, kp=2.0):
    o = detect(D, SE, N, kp=kp)
    if o["status"] in ("found", "capped"):
        return o["status"], o["tau"]
    if o["status"] == "unresolved":
        return "unresolved", max(o["tau_lb"], 600.0)
    return o["status"], 600.0


RULES = {"peak and reversal tests": cd.plan_detect,
         "VM06": lambda D, SE, N: adopted(D, SE, N, kp=0.0),
         "VM06 + peak test": adopted}
aligned = lambda p: {**p, "aw": -p["aw"]}
SCEN = {
    "MESOSCALE stress, opposing": MESO_STRESS,
    "MESOSCALE stress, aligned": aligned(MESO_STRESS),
    "MESOSCALE typical": MESO_TYPICAL,
    "MESOSCALE typical, T = 5 s": {**MESO_TYPICAL, "T": 5.0},
    "MESOSCALE typical, zero flux": {**MESO_TYPICAL, "rho": 0.0},
    "STABLE stress, opposing": STABLE_STRESS,
    "STABLE stress, aligned": aligned(STABLE_STRESS),
    "STABLE typical": STABLE_TYPICAL,
    "STABLE typical, weak (rho = -0.1)": {**STABLE_TYPICAL, "rho": -0.1},
    "STABLE typical, very weak (rho = -0.05)": {**STABLE_TYPICAL, "rho": -0.05},
    "STABLE typical, zero flux": {**STABLE_TYPICAL, "rho": 0.0},
}
for name, kw in SCEN.items():
    true = kw["rho"] * kw["sw"] * kw["su"]
    res = {r: [] for r in RULES}
    for s in range(50):
        D, SE, N, u, w, u_t, w_t, slot0 = cd.realization(s, **kw)
        for r, fn in RULES.items():
            st, tau = fn(D, SE, N)
            res[r].append((st, tau, cd.flux(u, w, tau, slot0), cd.flux(u_t, w_t, tau, slot0)))
    print(f"\n{name}  (turbulent w'u' {true:+.5f})")
    for r in RULES:
        st = [x[0] for x in res[r]]; taus = np.array([x[1] for x in res[r]])
        fs = np.array([x[2] for x in res[r]]); ft = np.array([x[3] for x in res[r]])
        v, c = np.unique(taus, return_counts=True)
        line = f"  {r:24s} weak {st.count('weak'):2d}/50; tau {dict(zip(v.tolist(), c.tolist()))}"
        if true != 0:
            line += (f"\n  {'':24s} mean sel/true {np.mean(fs / true):.3f}; "
                     f"within 10% of turbulence-only {np.mean(np.abs(fs / ft - 1) < 0.1):.0%}")
        print(line)
