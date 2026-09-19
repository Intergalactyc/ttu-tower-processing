import numpy as np, time
HZ = 50
def hann(width_s):
    H = int(round(width_s * HZ / 2)); j = np.arange(-(H - 1), H)
    return np.cos(np.pi * j / (2 * H)) ** 2               # zero at +-H, H-1 samples of reach
def resp(k, f):                                           # |H(f)| of a normalized symmetric kernel
    j = np.arange(len(k)) - (len(k) - 1) / 2
    return np.abs(np.sum(k[None] * np.cos(2 * np.pi * f[:, None] * j[None] / HZ), 1)) / k.sum()
f = np.linspace(1e-4, 5, 200001)
for tr in (10.0, 1.0):
    k = hann(2 * tr); b = np.ones(int(tr * HZ) + 1)
    rk, rb = resp(k, f), resp(b, f)
    fc_k = f[np.argmax(rk < 2 ** -0.5)]; fc_b = f[np.argmax(rb < 2 ** -0.5)]
    fs = 1 / (2 * np.pi * tr)
    side_k = 20 * np.log10(rk[f > 2 / (2 * tr)].max()); side_b = 20 * np.log10(rb[f > 1 / tr].max())
    print(f"tau_r={tr:g}s: Hann(2tau_r) L={len(k)} -3dB {fc_k:.4f} Hz (period {1/fc_k:.1f} s), "
          f"at sensor f_c={fs:.4f}: {resp(k, np.array([fs]))[0]:.3f}, max sidelobe {side_k:.1f} dB | "
          f"boxcar(tau_r) -3dB {fc_b:.4f} Hz, at sensor f_c {resp(b, np.array([fs]))[0]:.3f}, sidelobe {side_b:.1f} dB")

def smooth_direct(x, ok, k, c):
    x0 = np.where(ok, x, 0.0)
    S = np.convolve(x0, k, "same"); W = np.convolve(ok.astype(float), k, "same")
    return S, W
def smooth_fft(x, ok, k, c):
    x0 = np.where(ok, x, 0.0); n = len(x) + len(k) - 1; m = 1 << (n - 1).bit_length()
    K = np.fft.rfft(k, m); h = (len(k) - 1) // 2
    S = np.fft.irfft(np.fft.rfft(x0, m) * K, m)[h:h + len(x)]
    W = np.fft.irfft(np.fft.rfft(ok.astype(float), m) * K, m)[h:h + len(x)]
    return S, W
rng = np.random.default_rng(0)
n = 150000
x = 290 + np.cumsum(rng.standard_normal(n)) * 1e-3; x = np.round(x / 0.002) * 0.002
ok = np.ones(n, bool); ok[40000:40200] = False; x[~ok] = np.nan
k = hann(20.0)
for fn in (smooth_direct, smooth_fft):
    t0 = time.perf_counter()
    for _ in range(5): S, W = fn(x, ok, k, 0.75)
    print(fn.__name__, f"{(time.perf_counter() - t0) / 5 * 1000:.1f} ms per variable (L={len(k)})")
Sd, Wd = smooth_direct(x, ok, k, 0.75); Sf, Wf = smooth_fft(x, ok, k, 0.75)
core = slice(30000, 120000)
print("fft vs direct, max rel diff of S/W over core:", np.nanmax(np.abs(Sf[core] / Wf[core] - Sd[core] / Wd[core]) / 290))
# seam invariance: same core computed from a longer span
x2 = np.concatenate((290 + np.zeros(77777), x, 290 + np.zeros(33333))); ok2 = np.concatenate((np.ones(77777, bool), ok, np.ones(33333, bool)))
S2, W2 = smooth_fft(x2, ok2, k, 0.75)
a = Sf[core] / Wf[core]; b = (S2 / W2)[77777:77777 + n][core]
print("fft span A vs span B, max rel diff over core:", np.nanmax(np.abs(a - b) / np.abs(a)))
S3, W3 = smooth_direct(x2, ok2, k, 0.75); b3 = (S3 / W3)[77777:77777 + n][core]; a3 = Sd[core] / Wd[core]
print("direct span A vs span B, bitwise equal:", np.array_equal(a3[np.isfinite(a3)], b3[np.isfinite(b3)]))
# step size of the output: max sample-to-sample change vs the input's quantization step
xs = a
print("max |diff| of smoothed series:", np.nanmax(np.abs(np.diff(xs))), "(input step 0.002)")
