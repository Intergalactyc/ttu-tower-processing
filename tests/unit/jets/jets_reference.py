"""A slot-by-slot reading of the nose rules, to check the vectorized algorithm against."""
import numpy as np

from ttu_tower.jets.identify import NoseParams


def reference_nose(u, params: NoseParams):
    booms = list(params.profile_booms)
    v = [u[b - 1] for b in booms]
    ok = [np.isfinite(x) for x in v]
    if not all(np.isfinite(u[b - 1]) for b in params.required_booms):
        return "no_data", None
    candidates = []
    for i in range(1, len(v) - 1):
        if ok[i - 1] and ok[i] and ok[i + 1] and v[i] > v[i - 1] and v[i] > v[i + 1]:
            fall = v[i] - min(v[k] for k in range(i + 1, len(v)) if ok[k])
            candidates.append((i, v[i], fall, fall / v[i]))
    if not candidates:
        return "no_peak", None
    fast = [c for c in candidates if c[1] >= params.min_nose_speed]
    if not fast:
        return "too_slow", booms[max(candidates, key=lambda c: (c[1], -c[0]))[0]]

    def strong(c):
        return ((params.min_falloff_abs is None or c[2] >= params.min_falloff_abs)
                and (params.min_falloff_rel is None or c[3] >= params.min_falloff_rel))
    qualifying = [c for c in fast if strong(c)]
    if not qualifying:
        return "weak_falloff", booms[max(fast, key=lambda c: (c[2], -c[0]))[0]]
    return "jet", booms[max(qualifying, key=lambda c: (c[1], -c[0]))[0]]
