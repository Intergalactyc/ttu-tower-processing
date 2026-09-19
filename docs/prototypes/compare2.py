import numpy as np
import compare_detect as cd
S = cd.S
def smooth14(D, SE):
    # 1-2-1 over modes 1..14 (mode 15 has one pair and is left out), edges replicated
    D14, S14 = D[:14], SE[:14]
    Dp = np.concatenate(([D14[0]], D14, [D14[-1]])); Sp = np.concatenate(([S14[0]], S14, [S14[-1]]))
    Ds = (Dp[:-2] + 2*Dp[1:-1] + Dp[2:]) / 4
    Ss = np.sqrt(Sp[:-2]**2 + 4*Sp[1:-1]**2 + Sp[2:]**2) / 4
    return np.concatenate((Ds, D[14:])), np.concatenate((Ss, SE[14:]))

def vm06w(D, SE, N, kp=2.0, i_start=2, i_min=7, i_max=14):
    Ds, Ss = smooth14(D, SE)
    p = None
    for i in range(i_start, i_max):
        if np.sign(Ds[i]) != np.sign(Ds[i-1]) or abs(Ds[i]) < abs(Ds[i-1]): p = i; break
    if p is None: p = i_max
    if abs(Ds[p-1]) < kp * Ss[p-1]:
        # first local max not significant: look for the first significant local max further up
        q = None
        for i in range(p, i_max + 1):
            is_max = i == i_max or abs(Ds[i]) < abs(Ds[i-1]) or np.sign(Ds[i]) != np.sign(Ds[i-1])
            if is_max and abs(Ds[i-1]) >= kp * Ss[i-1]: q = i; break
        if q is None: return "weak", 600.0
        p = q
    if p == i_max: return "capped", S[i_max-1]
    sg = np.sign(Ds[p-1])
    for j in range(p, i_max):
        if np.sign(Ds[j]) != sg or abs(Ds[j]) > abs(Ds[j-1]): return "found", max(S[j], S[i_min-1])
    return "capped", S[i_max-1]

def hybrid(D, SE, N):
    Ds, Ss = smooth14(D, SE)
    return cd.plan_detect(Ds, Ss, N, i_start=2)

ALGS = {"plan": lambda D, SE, N: cd.plan_detect(D, SE, N), "vm06": lambda D, SE, N: cd.vm06_detect(D),
        "vm06w": vm06w, "hybrid": hybrid}
SCEN = {
    "S3 incoherent meso": dict(T=1.0, sw=0.5, su=1.0, rho=-0.35, au=1.0, aw=0.05, shared=False),
    "S4 weak turbulence": dict(T=1.0, sw=0.3, su=0.6, rho=-0.1, au=1.0, aw=0.05, shared=True),
    "S6 no mesoscale": dict(T=1.0, sw=0.5, su=1.0, rho=-0.35, au=0.0, aw=0.0, shared=True),
    "S7 long T=5s, rho=-0.15, incoherent": dict(T=5.0, sw=0.5, su=1.0, rho=-0.15, au=1.0, aw=0.05, shared=False),
    "S8 zero flux, incoherent meso": dict(T=1.0, sw=0.5, su=1.0, rho=0.0, au=1.0, aw=0.05, shared=False),
    "S9 very weak rho=-0.05, incoherent": dict(T=1.0, sw=0.5, su=1.0, rho=-0.05, au=1.0, aw=0.05, shared=False),
}
R = 40
for name, kw in SCEN.items():
    true = kw["rho"] * kw["sw"] * kw["su"]
    res = {a: [] for a in ALGS}
    for s in range(R):
        D, SE, N, u, w, u_t, w_t, slot0 = cd.realization(s, **kw)
        for a, fn in ALGS.items():
            st, tau = fn(D, SE, N)
            res[a].append((st, tau, cd.flux(u, w, tau, slot0), cd.flux(u_t, w_t, tau, slot0)))
    print(f"\n{name}  (true flux {true:.4f})")
    for a in ALGS:
        st = [r[0] for r in res[a]]; taus = np.array([r[1] for r in res[a]])
        fs = np.array([r[2] for r in res[a]]); ft = np.array([r[3] for r in res[a]])
        vals, cnt = np.unique(taus, return_counts=True)
        line = f"  {a:7s} {dict((k, st.count(k)) for k in sorted(set(st)))}  tau {dict((float(v), int(c)) for v, c in zip(vals, cnt))}"
        if true != 0:
            line += f"\n          mean sel/true {np.mean(fs/true):.3f}  turb-at-sel/true {np.mean(ft/true):.3f}  |sel/turb-1|<.1: {np.mean(np.abs(fs/ft-1)<0.1):.2f}"
        else:
            line += f"\n          mean |sel flux| {np.mean(np.abs(fs)):.4f} (vs sigma_w*sigma_u = {kw['sw']*kw['su']:.2f})"
        print(line)
