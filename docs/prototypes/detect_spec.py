"""Prototype of the decided detection rule (VM06 + peak significance), written to match the
plan.md Phase 7 specification line by line. Modes are 1-based in comments, 0-based in arrays."""
import numpy as np

P = 0.29296875 * 2.0 ** np.arange(15)          # P_1..P_15 (VM06 labeling)
NBLK = 2.0 ** (15 - np.arange(1, 16))


def detect(D, SE, N, has_data=True, kp=2.0, min_scale=0.5, min_pairs=2, min_tau=9.375,
           i_top=14, smoothing="121"):
    D, SE, N = map(lambda a: np.asarray(a, float), (D, SE, N))
    out = dict(status=None, tau=np.nan, tau_lb=np.nan, sign=0, peak=np.nan, rev=np.nan, rtype=None)
    if not has_data:
        out["status"] = "no_data"; return out
    lo = int(np.argmax(P >= min_scale)) + 1                     # first analysed mode (2)
    usable = (N >= min_pairs) & np.isfinite(SE)
    u = next((i for i in range(lo, i_top + 1) if not usable[i - 1]), i_top + 1)
    R = list(range(lo, u))                                      # analysis range lo..u-1
    Ds, Ss = {}, {}
    for i in R:
        if smoothing == "none":
            Ds[i], Ss[i] = D[i - 1], SE[i - 1]; continue
        js = [(j, w) for j, w in ((i - 1, 1.0), (i, 2.0), (i + 1, 1.0)) if j in R]
        ws = sum(w for _, w in js)
        Ds[i] = sum(w * D[j - 1] for j, w in js) / ws
        Ss[i] = np.sqrt(sum((w * SE[j - 1]) ** 2 for j, w in js)) / ws
    sgn = lambda x: float(np.sign(x))
    sig = lambda i: Ds[i] != 0 and abs(Ds[i]) >= kp * Ss[i]
    turn = lambda i: sgn(Ds[i]) != sgn(Ds[i - 1]) or abs(Ds[i]) < abs(Ds[i - 1])  # i-1 is a local max
    unresolved = lambda: {**out, "status": "unresolved", "tau_lb": P[u - 2]}
    # peak: the first local maximum (VM06: first decrease in magnitude or sign change) that is significant
    p = None
    for i in R[1:]:
        if turn(i) and sig(i - 1):
            p = i - 1; break
    if p is None:
        if not R or u <= i_top: return unresolved()
        if sig(R[-1]): p = R[-1]                                 # still rising at the 40-min mode
        else: out["status"] = "weak"; return out
    s = sgn(Ds[p]); out["sign"] = int(s); out["peak"] = P[p - 1]
    # gap: the next sign change or increase in magnitude after the peak
    for r in range(p + 1, R[-1] + 1):
        if sgn(Ds[r]) != s or abs(Ds[r]) > abs(Ds[r - 1]):
            out.update(status="found", tau=max(P[r - 2], min_tau), rev=P[r - 1],
                       rtype="sign" if sgn(Ds[r]) != s else "increase")
            return out
    if u <= i_top: return unresolved()
    out.update(status="capped", tau=P[i_top - 2])
    if N[14] >= 1 and np.isfinite(D[14]):                        # unconfirmed look at the 80-min mode
        if sgn(D[14]) != s: out.update(rev=P[14], rtype="sign")
        elif p < i_top and abs(D[14]) > abs(Ds[i_top]): out.update(rev=P[14], rtype="increase")
    return out


def fmt(o):
    s = o["status"]
    if s in ("found",): return f"found, tau={o['tau']:g}, rev@{o['rev']:g} ({o['rtype']}), peak@{o['peak']:g}, sign={o['sign']}"
    if s == "capped": return f"capped, tau={o['tau']:g}, diag rev={o['rev']} {o['rtype']}, sign={o['sign']}"
    if s == "unresolved": return f"unresolved, tau_lb={o['tau_lb']:g}"
    return s


if __name__ == "__main__":
    full = NBLK.copy()
    se = lambda v=0.05: np.full(15, v)
    cases = {
        "clean gap": [0.1, 0.3, 0.6, 1.0, 1.4, 1.6, 1.5, 1.1, 0.6, 0.2, -0.3, 0.8, -1.2, 2.0, 3.0],
        "dip before peak (SE .1)": ([0.1, 0.5, 1.0, 0.9, 1.6, 1.8, 1.2, 0.5, -0.2, 0.3, 0.1, 0.2, 0.1, 0.3, 0.2], 0.1),
        "increase-type": [0.2, 0.5, 0.9, 1.3, 1.8, 2.0, 1.4, 0.8, 0.5, 0.7, 1.2, 2.0, 0.5, 1.0, 1.0],
        "sign right after peak": [0.1, 0.4, 0.9, 1.4, 1.8, 2.0, 1.9, -1.5, -2.5, -1.0, 0.2, 0.5, 0.3, 0.2, 0.1],
        "weak": [0.05, -0.04, 0.08, 0.02, 0.09, -0.05, 0.03, 0, 0.07, -0.02, 0.01, 0.05, -0.03, 0.02, 0.01],
        "insignificant bump first": [0.0, 0.05, 0.12, 0.05, 0.0, 0.3, 0.8, 1.5, 2.0, 1.5, 0.6, -0.2, 0.3, 0.2, 0.1],
        "mode 1 ignored": [5.0, 0.3, 0.6, 1.0, 1.4, 1.6, 1.5, 1.1, 0.6, 0.2, -0.3, 0.8, -1.2, 2.0, 3.0],
        "capped": list(np.linspace(0.1, 3.0, 15)),
        "capped, 80-min reversal": list(np.linspace(0.1, 2.8, 14)) + [-1.0],
        "negative flux": list(-np.array([0.1, 0.3, 0.6, 1.0, 1.4, 1.6, 1.5, 1.1, 0.6, 0.2, -0.3, 0.8, -1.2, 2.0, 3.0])),
        "early reversal (clipped)": [0.5, 1.5, 2.0, 1.0, -0.4, 0.3, 0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.2, 0.1, 0.1],
        "SE = 0, flat": ([0.0] * 15, 0.0),
    }
    for name, c in cases.items():
        D, s = (c if isinstance(c, tuple) else (c, 0.05))
        o = detect(D, se(s), full)
        print(f"{name:28s} {fmt(o)}")
    # unresolved: capped array with N_12 = 1
    N = full.copy(); N[11] = 1
    print(f"{'unresolved (N_12=1)':28s} {fmt(detect(np.linspace(0.1, 3.0, 15), se(), N))}")
    N = full.copy(); N[11] = 1
    print(f"{'unresolved, gap before u':28s} {fmt(detect(cases['clean gap'], se(), N))}")
    print(f"{'no data':28s} {fmt(detect(cases['clean gap'], se(), full, has_data=False))}")
    print(f"{'SE=0, clean gap':28s} {fmt(detect(cases['clean gap'], se(0.0), full))}")
    # smoothed values for the documented cases
    for name in ("clean gap", "increase-type", "sign right after peak", "insignificant bump first"):
        D = np.array(cases[name])
        lo = 2
        R = list(range(lo, 15))
        sm = []
        for i in R:
            js = [(j, w) for j, w in ((i - 1, 1.0), (i, 2.0), (i + 1, 1.0)) if j in R]
            sm.append(sum(w * D[j - 1] for j, w in js) / sum(w for _, w in js))
        print(name, "smoothed modes 2..14:", np.round(sm, 3).tolist())
