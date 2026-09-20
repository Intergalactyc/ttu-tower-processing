"""Turbulence diagnostics computed from pooled flux products."""


def transport_efficiency(s_all: float, s_pos: float, s_neg: float) -> float:
    """Transport efficiency from pooled flux sums: S_all / S_pos if S_all > 0
    (the flux is upward-dominated), else S_all / S_neg. NaN if the chosen
    denominator is 0 (Salesky et al. 2017).
    """
    denom = s_pos if s_all > 0 else s_neg
    if denom == 0:
        return float("nan")
    return s_all / denom
