import numpy as np
from scipy.signal import lfilter
HZ = 50; DT = 1 / HZ
S = 0.146484375 * 2.0 ** np.arange(15)          # child (sibling) width of modes 1..15; parent = 2*S
NPAIR = 2.0 ** (15 - np.arange(1, 16))

def ou(n, T, rng):
    a = np.exp(-DT / T); e = rng.standard_normal(n) * np.sqrt(1 - a * a)
    return lfilter([1.0], [1.0, -a], e, zi=[a * rng.standard_normal()])[0]

def floor_sums(x, nb=32768, num=1875, den=256):
    g = np.arange(len(x)); k = (g * den) // num
    fk = (np.minimum((k + 1) * num, (g + 1) * den) - g * den) / den; fk1 = 1 - fk; k1 = np.minimum(k + 1, nb)
    s = np.bincount(k, fk * x, nb + 1)[:nb] + np.bincount(k1, fk1 * x, nb + 1)[:nb]
    w = np.bincount(k, fk, nb + 1)[:nb] + np.bincount(k1, fk1, nb + 1)[:nb]
    return s, w

def modes(sx, sy, n):
    val = np.empty(15); se = np.empty(15); npr = np.empty(15)
    for i in range(1, 16):
        w = 2 ** (i - 1)
        ax = sx.reshape(-1, w).sum(1); ay = sy.reshape(-1, w).sum(1); an = n.reshape(-1, w).sum(1)
        mx, my = ax / an, ay / an
        p = (mx[1::2] - mx[0::2]) * (my[1::2] - my[0::2]) / 4
        val[i - 1] = p.mean(); npr[i - 1] = p.size
        se[i - 1] = p.std(ddof=1) / np.sqrt(p.size) if p.size >= 2 else np.nan
    return val, se, npr

def plan_detect(D, SE, N, kp=2.0, kr=2.0, i_min=7, i_max=14, i_start=1):
    usable = lambda i: N[i-1] >= 2 and np.isfinite(SE[i-1])
    cand = [i for i in range(i_start, i_max + 1) if usable(i)]
    z = lambda i: (D[i-1] / SE[i-1]) if SE[i-1] > 0 else (np.inf * np.sign(D[i-1]) if D[i-1] != 0 else 0.0)
    sig = [i for i in cand if abs(z(i)) >= kp]
    if not sig: return "weak", 600.0
    pairs = [i for i in sig if (i + 1) in sig and np.sign(D[i-1]) == np.sign(D[i])]
    sg = np.sign(D[pairs[0]-1]) if pairs else np.sign(D[max(sig, key=lambda i: (abs(z(i)), -i))-1])
    f = min(i for i in cand if sg * D[i-1] >= kp * SE[i-1])
    comb = lambda a, b: np.hypot(SE[a-1], SE[b-1])
    p = f; state = 0; m = None
    for i in range(f + 1, i_max + 1):
        if not usable(i): return "unresolved", max(S[i-1], 600.0)
        if state == 0:
            if sg * D[i-1] > sg * D[p-1]: p = i; continue
            if sg * (D[p-1] - D[i-1]) <= kr * comb(p, i): continue
            if sg * D[i-1] <= 0: return "found", max(S[i-1], S[i_min-1])
            state = 1; m = i; continue
        if sg * D[i-1] <= 0: return "found", max(S[i-1], S[i_min-1])
        if sg * D[i-1] < sg * D[m-1]: m = i; continue
        if sg * (D[i-1] - D[m-1]) > kr * comb(m, i): return "found", max(S[i-1], S[i_min-1])
    return "capped", S[i_max-1]

def smooth121(D, SE=None):
    Dp = np.concatenate(([D[0]], D, [D[-1]]))
    Ds = (Dp[:-2] + 2 * Dp[1:-1] + Dp[2:]) / 4
    if SE is None: return Ds
    Sp = np.concatenate(([SE[0]], SE, [SE[-1]]))
    return Ds, np.sqrt(Sp[:-2]**2 + 4 * Sp[1:-1]**2 + Sp[2:]**2) / 4

def vm06_detect(D, i_start=2, i_min=7, i_max=14):
    Ds = smooth121(D)
    p = None
    for i in range(i_start, i_max):
        if np.sign(Ds[i]) != np.sign(Ds[i-1]) or abs(Ds[i]) < abs(Ds[i-1]): p = i; break
    if p is None: return "capped", S[i_max-1]
    sg = np.sign(Ds[p-1])
    for j in range(p, i_max):
        if np.sign(Ds[j]) != sg or abs(Ds[j]) > abs(Ds[j-1]): return "found", max(S[j], S[i_min-1])
    return "capped", S[i_max-1]

def hybrid_detect(D, SE, N):
    Ds, SEs = smooth121(D, SE)
    return plan_detect(Ds, SEs, N, i_start=2)

ALGS = {"plan": lambda D, SE, N: plan_detect(D, SE, N),
        "vm06": lambda D, SE, N: vm06_detect(D),
        "hybrid": hybrid_detect}

def realization(seed, T, sw, su, rho, au, aw, shared, periods=(20, 30, 45, 60)):
    rng = np.random.default_rng(seed); n = 3 * 3600 * HZ
    x1, x2 = ou(n, T, rng), ou(n, T, rng)
    w_t = sw * x1; u_t = su * (rho * x1 + np.sqrt(1 - rho**2) * x2)
    t = np.arange(n) * DT
    ph = rng.uniform(0, 2*np.pi, len(periods)); phw = ph if shared else rng.uniform(0, 2*np.pi, len(periods))
    u = 8.0 + u_t + sum(au * np.sin(2*np.pi*t/(P*60) + q) for P, q in zip(periods, ph))
    w = w_t + sum(aw * np.sin(2*np.pi*t/(P*60) + q) for P, q in zip(periods, phw))
    slot0 = n // 2 - 15000; w0 = slot0 - 35*60*HZ
    su_, n_ = floor_sums(u[w0:w0 + 240000]); sw_, _ = floor_sums(w[w0:w0 + 240000])
    D, SE, N = modes(su_, sw_, n_)
    return D, SE, N, u, w, u_t, w_t, slot0

def flux(uu, ww, tau, slot0):
    L = int(round(tau * HZ))
    a, b = (slot0, slot0 + 30000) if tau <= 600 else (slot0 - 15000, slot0 + 45000)
    su, sw = uu[a:b], ww[a:b]; nb = len(su) // L
    bu, bw = su[:nb*L].reshape(nb, L), sw[:nb*L].reshape(nb, L)
    return np.mean(((bu - bu.mean(1, keepdims=True)) * (bw - bw.mean(1, keepdims=True))).mean(1))

SCEN = {
    "S1 coherent meso (aw=0.1)":        dict(T=1.0, sw=0.5, su=1.0, rho=-0.35, au=1.0, aw=0.1, shared=True),
    "S2 strong coherent meso (aw=0.3)": dict(T=1.0, sw=0.5, su=1.0, rho=-0.35, au=1.0, aw=0.3, shared=True),
    "S3 incoherent meso (aw=0.05)":     dict(T=1.0, sw=0.5, su=1.0, rho=-0.35, au=1.0, aw=0.05, shared=False),
    "S4 weak turbulence (rho=-0.1)":    dict(T=1.0, sw=0.3, su=0.6, rho=-0.1, au=1.0, aw=0.05, shared=True),
    "S5 stable, T=0.25s, tiny flux":    dict(T=0.25, sw=0.1, su=0.3, rho=-0.3, au=0.5, aw=0.02, shared=True),
    "S6 no mesoscale":                  dict(T=1.0, sw=0.5, su=1.0, rho=-0.35, au=0.0, aw=0.0, shared=True),
}
R = 40
for name, kw in SCEN.items():
    true = kw["rho"] * kw["sw"] * kw["su"]
    res = {a: [] for a in ALGS}
    for s in range(R):
        D, SE, N, u, w, u_t, w_t, slot0 = realization(s, **kw)
        for a, fn in ALGS.items():
            st, tau = fn(D, SE, N)
            res[a].append((st, tau, flux(u, w, tau, slot0), flux(u_t, w_t, tau, slot0)))
    f20 = np.mean([flux(*realization(s, **kw)[3:5], 1200, realization(s, **kw)[7]) for s in range(0)]) if False else None
    print(f"\n{name}  (true flux {true:.4f})")
    for a in ALGS:
        st = [r[0] for r in res[a]]; taus = np.array([r[1] for r in res[a]])
        fs = np.array([r[2] for r in res[a]]); ft = np.array([r[3] for r in res[a]])
        vals, cnt = np.unique(taus, return_counts=True)
        print(f"  {a:7s} status {dict((k, st.count(k)) for k in sorted(set(st)))}  tau {dict((float(v), int(c)) for v, c in zip(vals, cnt))}")
        print(f"          mean ratio sel/true {np.mean(fs/true):.3f}   turb-only-at-sel/true {np.mean(ft/true):.3f}   frac |sel/turbonly-1|<0.1 {np.mean(np.abs(fs/ft-1)<0.1):.2f}")
