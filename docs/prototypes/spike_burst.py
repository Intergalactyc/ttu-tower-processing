"""A burst of alternating extreme values in autocorrelated (OU, T = 1 s) data: how the run-length
spike rule classifies it, and how splitting outlier runs at large jumps would."""
import numpy as np
from scipy.signal import lfilter

HZ, Z, MAXRUN = 50, 3.5, 3


def ou(n, T, rng):
    a = np.exp(-1 / (HZ * T))
    return lfilter([1.0], [1.0, -a], rng.standard_normal(n) * np.sqrt(1 - a * a))


def runs(mask):
    d = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def classify(x, split_jumps):
    m = np.median(x); mad = np.median(np.abs(x - m))
    out = 0.6745 * np.abs(x - m) / mad > Z
    spikes, excursions = [], []
    for a, b in runs(out):
        cuts = [a]
        if split_jumps:
            jump = 0.6745 * np.abs(np.diff(x[a:b])) / mad > Z
            cuts += list(a + 1 + np.flatnonzero(jump))
        for s, e in zip(cuts, cuts[1:] + [b]):
            (spikes if e - s <= MAXRUN else excursions).append((s, e))
    return spikes, excursions


def kurt(x):
    d = x - x.mean()
    return np.mean(d**4) / np.mean(d**2) ** 2


rng = np.random.default_rng(1)
x = ou(30000, 1.0, rng)                                       # one 10-min slot, sigma = 1
x /= np.median(np.abs(x - np.median(x)))                      # scale so that MAD = 1
burst = np.array([-10, 10, 8, -7, 15, -12], float)
y = x.copy(); y[15000:15006] = burst
print(f"clean: kurtosis {kurt(x):.2f}; with burst: kurtosis {kurt(y):.2f}, "
      f"variance +{100 * (y.var() / x.var() - 1):.1f}%")
for split in (False, True):
    sp, ex = classify(y, split)
    hit = [r for r in sp + ex if r[0] < 15006 and r[1] > 15000]
    kind = ["spike" if r in sp else "excursion" for r in hit]
    lens = sorted(e - s for s, e in ex)
    print(f"{'split at jumps' if split else 'run length only':16s}: burst -> {list(zip(hit, kind))}; "
          f"other excursions kept: {len([r for r in ex if r not in hit])} (lengths {lens[:8]}...), "
          f"other spikes removed: {len([r for r in sp if r not in hit])}")


# A genuine intermittent burst (5 s, sigma 10x the quiet background, T = 0.3 s) in a quiet
# 5-min reference: how many of its samples each rule would remove as spikes.
def classify_sign(x):
    m = np.median(x); mad = np.median(np.abs(x - m))
    out = 0.6745 * np.abs(x - m) / mad > Z
    spikes = []
    for a, b in runs(out):
        side = np.sign(x[a:b] - m)
        cuts = [a] + list(a + 1 + np.flatnonzero(side[1:] != side[:-1]))
        spikes += [(s, e) for s, e in zip(cuts, cuts[1:] + [b]) if e - s <= MAXRUN]
    return spikes


removed = {"run length only": [], "split at jumps": [], "split at side changes": []}
for seed in range(200):
    rng = np.random.default_rng(seed)
    q = ou(15000, 1.0, rng)
    q[7400:7650] += 10 * ou(250, 0.3, rng)
    for name, sp in (("run length only", classify(q, False)[0]), ("split at jumps", classify(q, True)[0]),
                     ("split at side changes", classify_sign(q))):
        removed[name].append(sum(e - s for s, e in sp if 7400 <= s < 7650))
print("burst samples removed as spikes (of 250), mean / max over 200 bursts:")
for name, r in removed.items():
    print(f"  {name:22s} {np.mean(r):.1f} / {max(r)}")
