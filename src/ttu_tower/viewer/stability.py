"""The stability classification the viewer uses: five classes split by four
boundaries of one variable - Ri_b between two booms, or ζ = z/L at one boom.
It starts as the run's own (`post.stability`) and can be changed in the viewer;
the run's outputs are never changed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import pyarrow.dataset as pa_dataset

from ttu_tower.post.classify import IntervalClassifier, _parse_interval

NAMES = ("strongly unstable", "unstable", "neutral", "stable", "strongly stable")
VARIABLES = {"rib": "Ri_b", "zeta": "ζ = z/L"}
ZETA_EDGES = (-0.5, -0.05, 0.05, 0.5)  # a starting point when switching to ζ, not a standard


@dataclass(frozen=True)
class StabilitySpec:
    variable: str = "rib"  # "rib" (between booms[0] and booms[1]) or "zeta" (at booms[0])
    booms: tuple[int, int] = (2, 4)
    variant: str = "mrd"  # ζ's; Ri_b has none
    edges: tuple[float, float, float, float] = (-0.05, -0.01, 0.01, 0.1)

    def errors(self) -> list[str]:
        errors = []
        if self.variable not in VARIABLES:
            errors.append(f"no stability variable {self.variable!r}")
        if self.variable == "rib" and self.booms[0] == self.booms[1]:
            errors.append("Ri_b needs two different booms")
        if len(self.edges) != 4 or any(not np.isfinite(e) for e in self.edges):
            errors.append("four finite boundaries are needed")
        elif any(b <= a for a, b in zip(self.edges, self.edges[1:])):
            errors.append("the boundaries must increase")
        return errors

    def classes(self) -> list[tuple[str, str]]:
        """(name, interval) per class: below the first boundary, between each pair, above the last."""
        bounds = (-np.inf, *self.edges, np.inf)
        return [(name, f"[{lo},{hi})" if np.isfinite(lo) else f"(-inf,{hi})")
                for name, lo, hi in zip(NAMES, bounds[:-1], bounds[1:])]

    def describe(self) -> str:
        if self.variable == "rib":
            a, b = sorted(self.booms)
            return f"Ri_b(b{a}, b{b})"
        return f"ζ at b{self.booms[0]} ({self.variant})"

    def to_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, values: dict) -> StabilitySpec:
        return cls(variable=values.get("variable", "rib"), booms=tuple(values.get("booms", (2, 4))),
                   variant=values.get("variant", "mrd"), edges=tuple(float(e) for e in values.get("edges", ())))


def from_config(cfg_post) -> StabilitySpec | None:
    """The run's classification as a spec, or None where its classes aren't the
    five names as contiguous intervals (then the run's own is used as it is).
    """
    stability = cfg_post.stability
    names = [name for name, _ in stability.classes]
    if tuple(names) != NAMES:
        return None
    intervals = [_parse_interval(interval) for _, interval in stability.classes]
    if intervals[0].low != -np.inf or intervals[-1].high != np.inf:
        return None
    if any(a.high != b.low for a, b in zip(intervals, intervals[1:])):
        return None
    spec = StabilitySpec("rib", tuple(stability.pair), "mrd", tuple(float(i.high) for i in intervals[:-1]))
    if spec.classes() != [(n, _canonical(i)) for n, i in stability.classes]:
        return None  # e.g. other open/closed ends than [lo, hi)
    return spec


def _canonical(interval: str) -> str:
    """An interval string as `classes()` writes it."""
    i = _parse_interval(interval)
    lo = "-inf" if i.low == -np.inf else repr(float(i.low))
    hi = "inf" if i.high == np.inf else repr(float(i.high))
    return f"{'[' if i.low_inclusive else '('}{lo},{hi}{']' if i.high_inclusive else ')'}"


def values(index, spec: StabilitySpec) -> pd.Series:
    """The classifying variable per slot."""
    f = pa_dataset.field
    if spec.variable == "rib":
        a, b = sorted(spec.booms)
        rows = index.read("pairs", columns=["slot", "value"],
                          filter=(f("variable") == "rib") & (f("boom") == a) & (f("boom2") == b))
    else:
        boom = spec.booms[0]
        rows = index.read("boom_final", booms=[boom], columns=["slot", "value"],
                          filter=(f("variant") == spec.variant) & (f("variable") == "zeta") & f("stat").is_null()
                          & (f("boom") == boom))
    return rows.groupby("slot")["value"].first().sort_index()


def classify(series: pd.Series, classes) -> pd.Series:
    """slot -> class name (None for NaN or no class)."""
    names = IntervalClassifier(list(classes)).classify(series.to_numpy())
    return pd.Series(names, index=series.index, name="stability_class")
