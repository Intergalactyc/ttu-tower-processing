import numpy as np
from detect_spec import detect, fmt, NBLK
full = NBLK.copy(); se = lambda v=0.05: np.full(15, v)
C = {
 "sign-type gap": [0.2, 0.5, 0.9, 1.3, 1.6, 1.7, 1.5, 1.0, 0.4, -0.3, -0.6, -0.4, -0.2, 0.3, 0.5],
 "increase-type gap": [0.2, 0.5, 0.9, 1.3, 1.6, 1.7, 1.4, 0.9, 0.5, 0.3, 0.4, 0.8, 1.4, 2.0, 2.5],
 "wobble before peak (SE .1)": ([0.1, 0.5, 1.0, 0.9, 1.6, 1.8, 1.5, 1.0, 0.4, -0.2, -0.4, -0.2, 0.1, 0.2, 0.1], 0.1),
 "insignificant bump first": [0.0, 0.02, 0.06, 0.02, 0.0, 0.3, 0.8, 1.5, 2.0, 1.5, 0.6, -0.2, -0.3, -0.1, 0.1],
 "weak": [0.05, -0.04, 0.08, 0.02, 0.09, -0.05, 0.03, 0, 0.07, -0.02, 0.01, 0.05, -0.03, 0.02, 0.01],
 "mode 1 ignored": [-5.0, 0.5, 0.9, 1.3, 1.6, 1.7, 1.5, 1.0, 0.4, -0.3, -0.6, -0.4, -0.2, 0.3, 0.5],
 "capped (rising)": list(np.linspace(0.1, 3.0, 15)),
 "capped, 80-min sign": list(np.linspace(0.1, 2.8, 14)) + [-1.0],
 "capped, peak then flat decline, 80-min rise": [0.2, 0.5, 0.9, 1.3, 1.6, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 1.5],
 "negative flux": list(-np.array([0.2, 0.5, 0.9, 1.3, 1.6, 1.7, 1.5, 1.0, 0.4, -0.3, -0.6, -0.4, -0.2, 0.3, 0.5])),
 "early reversal (clipped)": [0.5, 1.5, 2.0, 1.0, -1.5, -0.3, 0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.2, 0.1, 0.1],
 "SE = 0, sign-type gap": ([0.2, 0.5, 0.9, 1.3, 1.6, 1.7, 1.5, 1.0, 0.4, -0.3, -0.6, -0.4, -0.2, 0.3, 0.5], 0.0),
 "SE = 0, all zero": ([0.0]*15, 0.0),
}
for name, c in C.items():
    D, s = c if isinstance(c, tuple) else (c, 0.05)
    print(f"{name:45s} {fmt(detect(D, se(s), full))}")
w = C["wobble before peak (SE .1)"][0]
print(f"{'wobble, smoothing=none':45s} {fmt(detect(w, se(0.1), full, smoothing='none'))}")
N = full.copy(); N[11] = 1
print(f"{'unresolved: capped with N_12=1':45s} {fmt(detect(C['capped (rising)'], se(), N))}")
print(f"{'sign-type gap with N_12=1':45s} {fmt(detect(C['sign-type gap'], se(), N))}")
N = full.copy(); N[1] = 1
print(f"{'N_2=1 (no usable range)':45s} {fmt(detect(C['sign-type gap'], se(), N))}")
print(f"{'no data':45s} {fmt(detect(C['sign-type gap'], se(), full, has_data=False))}")
