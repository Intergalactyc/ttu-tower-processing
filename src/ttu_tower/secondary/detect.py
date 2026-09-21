"""tau detection on one cospectrum of one (slot, boom, variant): Vickers &
Mahrt (2006), plus a significance test on the peak, and the same test on a
reversal only where accepting it would clip tau to the floor.
"""
from dataclasses import dataclass

import numpy as np

from ttu_tower.constants import SAMPLE_HZ
from ttu_tower.primary.mrd import mode_scales_s

STATUSES = ("no_data", "weak", "found", "capped", "unresolved")


@dataclass(frozen=True)
class Detection:
    status: str
    tau_s: float
    tau_lb_s: float
    sign: int
    peak_scale_s: float
    reversal_scale_s: float
    reversal_type: str | None


def _result(status: str, *, tau_s=np.nan, tau_lb_s=np.nan, sign: int = 0, peak_scale_s=np.nan,
            reversal_scale_s=np.nan, reversal_type: str | None = None) -> Detection:
    return Detection(status, float(tau_s), float(tau_lb_s), int(sign), float(peak_scale_s),
                      float(reversal_scale_s), reversal_type)


def detect(D: np.ndarray, SE: np.ndarray, N: np.ndarray, family_has_data: bool, cfg_det) -> Detection:
    """One cospectrum's status, tau (or its lower bound), sign and peak/reversal
    scales. `D`, `SE`, `N` are the mode 1..K rows of the `mrd` table for this
    (slot, boom, variant, spectrum), ordered by ascending scale. Mode scales
    are rederived from K = len(D), so the caller need not pass them.
    """
    if not family_has_data:
        return _result("no_data")

    D = np.asarray(D, dtype=np.float64)
    SE = np.asarray(SE, dtype=np.float64)
    N = np.asarray(N)
    scales = mode_scales_s(D.size, SAMPLE_HZ)  # P_i, i = 1..K, 0-indexed here

    def scale(i: int) -> float:  # 1-indexed mode -> P_i
        return float(scales[i - 1])

    usable = (N >= 2) & np.isfinite(SE)
    i_lo = int(np.flatnonzero(scales >= cfg_det.min_scale_s)[0]) + 1
    i_top = int(np.argmin(np.abs(scales - 2.0 * cfg_det.max_tau_s))) + 1

    u = i_top + 1
    for i in range(i_lo, i_top + 1):
        if not usable[i - 1]:
            u = i
            break

    lo, hi = i_lo, u - 1  # R = [lo, hi], inclusive; empty if hi < lo

    d_smooth: dict[int, float] = {}
    se_smooth: dict[int, float] = {}
    for i in range(lo, hi + 1):
        members = [(2, i)] + [(1, j) for j in (i - 1, i + 1) if lo <= j <= hi]
        wsum = sum(w for w, _ in members)
        d_smooth[i] = sum(w * D[j - 1] for w, j in members) / wsum
        se_smooth[i] = np.sqrt(sum((w * SE[j - 1]) ** 2 for w, j in members)) / wsum

    def significant(i: int) -> bool:
        return d_smooth[i] != 0 and abs(d_smooth[i]) >= cfg_det.peak_significance_se * se_smooth[i]

    p = None
    for m in range(lo, hi):  # m+1 must also be in R
        is_local_max = np.sign(d_smooth[m + 1]) != np.sign(d_smooth[m]) or abs(d_smooth[m + 1]) < abs(d_smooth[m])
        if is_local_max and significant(m):
            p = m
            break

    if p is None:
        if u <= i_top:
            return _result("unresolved", tau_lb_s=scale(u - 1))
        if hi in d_smooth and significant(hi):
            p = hi
        else:
            return _result("weak")

    sigma = int(np.sign(d_smooth[p]))

    # A reversal is accepted on sight almost everywhere - deliberately, so
    # contamination sharing the turbulent flux's sign gets cut off as soon as
    # it appears, rather than waiting for it to grow "significant". But right
    # at the floor, every candidate maps to the same clipped tau regardless
    # of how it's justified, and small scales are exactly where SE can shrink
    # faster than the real signal (huge per-mode sample counts), so a value
    # statistically indistinguishable from zero can register as a
    # "reversal" there. Only when accepting the candidate as-is would clip
    # tau to the floor, also require the same significance test the peak
    # already gets - elsewhere a reversal still needs no such test.
    reversal = None
    for r in range(p + 1, hi + 1):
        would_clip = scale(r - 1) <= cfg_det.min_tau_s + 1e-9
        if np.sign(d_smooth[r]) != sigma:
            if would_clip and not significant(r):
                continue
            reversal = (r, "sign")
            break
        if abs(d_smooth[r]) > abs(d_smooth[r - 1]):
            if would_clip and not significant(r):
                continue
            reversal = (r, "increase")
            break

    if reversal is not None:
        r, reversal_type = reversal
        tau_s = max(scale(r - 1), cfg_det.min_tau_s)
        return _result("found", tau_s=tau_s, sign=sigma, peak_scale_s=scale(p),
                        reversal_scale_s=scale(r), reversal_type=reversal_type)

    if u <= i_top:
        return _result("unresolved", tau_lb_s=scale(u - 1), sign=sigma, peak_scale_s=scale(p))

    # capped: no reversal through the 40-min mode. Diagnostic at the 80-min
    # mode (i_top + 1, one block, unsmoothed), if it exists and is valid.
    reversal_scale_s, reversal_type = np.nan, None
    diag_i = i_top + 1
    if diag_i <= D.size and N[diag_i - 1] >= 1 and np.isfinite(D[diag_i - 1]):
        d_diag = D[diag_i - 1]
        if np.sign(d_diag) != sigma:
            reversal_scale_s, reversal_type = scale(diag_i), "sign"
        elif p < i_top and abs(d_diag) > abs(d_smooth[i_top]):
            reversal_scale_s, reversal_type = scale(diag_i), "increase"

    return _result("capped", tau_s=cfg_det.max_tau_s, sign=sigma, peak_scale_s=scale(p),
                    reversal_scale_s=reversal_scale_s, reversal_type=reversal_type)
