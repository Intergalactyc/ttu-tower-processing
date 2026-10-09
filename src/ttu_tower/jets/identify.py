"""Low-level jet detection: wind-speed profiles in, one row per slot out.

Every algorithm has the same signature, `detect(profiles, params) -> DataFrame`,
and registers itself with a parameter dataclass whose fields describe
themselves (label, unit, kind, limits), so callers - the viewer's form, a
notebook, a later export - can list, configure and run any of them alike.

Output contract, shared by all algorithms (one row per slot of `profiles`):
`is_jet`, `status` (`jet`, `no_data`, or the algorithm's own rejection codes),
and the nose and falloff of the jet or of the best rejected candidate.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from ttu_tower.constants import BOOMS, HEIGHTS

JET = "jet"
NO_DATA = "no_data"  # every algorithm uses this for missing data: gap filling never bridges it
COLUMNS = ("is_jet", "status", "nose_boom", "z_nose", "u_nose", "z_min_above", "u_min_above",
           "falloff_abs", "falloff_rel")
KINDS = ("float", "optional_float", "int", "booms", "bool")


# --- profiles ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class Profiles:
    """Wind-speed profiles on a complete, gap-free slot grid."""

    slots: np.ndarray  # (n,) int64, consecutive
    booms: tuple[int, ...]  # (m,) ascending by height
    ws: np.ndarray  # (n, m) m/s, NaN = missing or filtered
    heights: np.ndarray | None = None  # (m,) m; None = the tower's
    extra: Mapping[str, np.ndarray] = field(default_factory=dict)  # (n, m) arrays, e.g. "wd"

    def __post_init__(self):
        slots = np.asarray(self.slots, dtype=np.int64)
        object.__setattr__(self, "slots", slots)
        object.__setattr__(self, "booms", tuple(int(b) for b in self.booms))
        object.__setattr__(self, "ws", np.asarray(self.ws, dtype=float))
        heights = self.heights
        heights = np.array([HEIGHTS[b] for b in self.booms]) if heights is None else np.asarray(heights, float)
        object.__setattr__(self, "heights", heights)
        n, m = len(slots), len(self.booms)
        if self.ws.shape != (n, m) or heights.shape != (m,):
            raise ValueError(f"ws must be (slots, booms) = ({n}, {m}), heights ({m},)")
        if n > 1 and np.any(np.diff(slots) != 1):
            raise ValueError("slots must be a complete, consecutive grid")
        if m > 1 and np.any(np.diff(heights) <= 0):
            raise ValueError("booms must be ordered by increasing height")
        for name, arr in self.extra.items():
            if np.shape(arr) != (n, m):
                raise ValueError(f"extra[{name!r}] must be (slots, booms)")

    @classmethod
    def from_frame(cls, ws: pd.DataFrame, extra: Mapping[str, pd.DataFrame] | None = None) -> Profiles:
        """From a slot-indexed frame with one column per boom; missing slots
        between the first and last become all-NaN rows.
        """
        booms = sorted(ws.columns, key=lambda b: HEIGHTS[b])
        grid = np.arange(ws.index.min(), ws.index.max() + 1) if len(ws) else np.array([], dtype=np.int64)
        frame = ws.reindex(index=grid, columns=booms)
        more = {k: v.reindex(index=grid, columns=booms).to_numpy(float) for k, v in (extra or {}).items()}
        return cls(grid, tuple(booms), frame.to_numpy(float), extra=more)

    def column(self, boom: int) -> int:
        return self.booms.index(boom)

    def subset(self, booms) -> Profiles:
        missing = [b for b in booms if b not in self.booms]
        if missing:
            raise ValueError(f"booms {missing} aren't in these profiles")
        cols = [self.column(b) for b in sorted(booms, key=lambda b: self.heights[self.column(b)])]
        return Profiles(self.slots, tuple(self.booms[c] for c in cols), self.ws[:, cols], self.heights[cols],
                        {k: v[:, cols] for k, v in self.extra.items()})


# --- parameters -------------------------------------------------------------------------------

def param(default, *, label: str, unit: str = "", kind: str = "float", lo=None, hi=None, help: str = "",
          suggested=None):
    """A parameter field that describes itself. `suggested` is the value an
    `optional_float` takes when switched on from off.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown parameter kind {kind!r}")
    if kind == "booms":
        default = tuple(default)
    meta = dict(label=label, unit=unit, kind=kind, lo=lo, hi=hi, help=help, suggested=suggested)
    return field(default=default, metadata=meta)


@dataclass(frozen=True)
class Params:
    """Base for every parameter set: range checks from the field metadata."""

    def validate(self) -> list[str]:
        errors = []
        for f in dataclasses.fields(self):
            meta, value = f.metadata, getattr(self, f.name)
            label = meta.get("label", f.name)
            if meta.get("kind") == "booms":
                bad = [b for b in value if b not in BOOMS]
                if bad:
                    errors.append(f"{label}: no boom {bad}")
                continue
            if meta.get("kind") == "bool":
                if not isinstance(value, bool):
                    errors.append(f"{label} must be on or off")
                continue
            if value is None:
                if meta.get("kind") != "optional_float":
                    errors.append(f"{label} is required")
                continue
            if meta.get("lo") is not None and value < meta["lo"]:
                errors.append(f"{label} must be at least {meta['lo']}")
            if meta.get("hi") is not None and value > meta["hi"]:
                errors.append(f"{label} must be at most {meta['hi']}")
        return errors


def describe(params_cls) -> list[dict]:
    """Each field's name, default and metadata, in declaration order (what a form is built from)."""
    return [dict(name=f.name, default=f.default, **f.metadata) for f in dataclasses.fields(params_cls)]


def to_dict(params) -> dict:
    return {f.name: (list(v) if isinstance(v := getattr(params, f.name), tuple) else v)
            for f in dataclasses.fields(params)}


def from_dict(params_cls, values: Mapping):
    """A parameter set from (possibly partial or stale) saved values: unknown keys
    are ignored, missing ones take their defaults.
    """
    kwargs = {}
    for f in dataclasses.fields(params_cls):
        if f.name in values:
            v = values[f.name]
            kwargs[f.name] = tuple(int(b) for b in v) if f.metadata.get("kind") == "booms" else v
    return params_cls(**kwargs)


# --- registry ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class Algorithm:
    key: str
    label: str
    params: type
    detect: Callable[[Profiles, Params], pd.DataFrame]
    description: str


ALGORITHMS: dict[str, Algorithm] = {}


def register(key: str, label: str, params: type):
    def decorate(fn):
        ALGORITHMS[key] = Algorithm(key, label, params, fn, " ".join((fn.__doc__ or "").split()))
        return fn
    return decorate


def detect(key: str, profiles: Profiles, params: Params | None = None) -> pd.DataFrame:
    """Run algorithm `key` (its defaults if `params` is None), checking the
    parameters first and the output contract after.
    """
    algorithm = ALGORITHMS[key]
    params = algorithm.params() if params is None else params
    errors = params.validate()
    if errors:
        raise ValueError("; ".join(errors))
    out = algorithm.detect(profiles, params)
    missing = [c for c in COLUMNS if c not in out.columns]
    if missing or len(out) != len(profiles.slots):
        raise RuntimeError(f"algorithm {key!r} broke the output contract (missing {missing})")
    return out


def result_frame(profiles: Profiles, status: np.ndarray, statuses: tuple[str, ...], nose: np.ndarray,
                 z_nose, u_nose, z_min, u_min) -> pd.DataFrame:
    """The contract's columns from per-slot arrays; `nose` is a column of
    `profiles` or -1 for none.
    """
    falloff = u_nose - u_min
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = falloff / u_nose
    nose_boom = pd.array(np.asarray(profiles.booms)[np.clip(nose, 0, None)], dtype="Int8")
    nose_boom[nose < 0] = pd.NA
    return pd.DataFrame({
        "is_jet": status == JET,
        "status": pd.Categorical(status, categories=statuses),
        "nose_boom": nose_boom,
        "z_nose": z_nose, "u_nose": u_nose, "z_min_above": z_min, "u_min_above": u_min,
        "falloff_abs": falloff, "falloff_rel": rel,
    }, index=pd.Index(profiles.slots, name="slot"))


# --- algorithm "nose" -------------------------------------------------------------------------

NOSE_STATUSES = (NO_DATA, "no_peak", "too_slow", "weak_falloff", JET)


@dataclass(frozen=True)
class NoseParams(Params):
    profile_booms: tuple[int, ...] = param(
        BOOMS, label="profile booms", kind="booms",
        help="The booms making up the profile; the peak and the falloff are found among them.")
    required_booms: tuple[int, ...] = param(
        (7, 8, 9, 10), label="required booms", kind="booms",
        help="Booms that must all be present and good, or the slot is skipped as no data.")
    min_nose_speed: float = param(
        10.0, label="min peak speed", unit="m/s", lo=0.0, hi=50.0,
        help="The peak speed must be at least this.")
    min_falloff_abs: float | None = param(
        2.0, label="min falloff", unit="m/s", kind="optional_float", lo=0.0, hi=30.0, suggested=2.0,
        help="The peak must be at least this much faster than the slowest boom above it.")
    min_falloff_rel: float | None = param(
        None, label="min relative falloff", unit="of peak speed", kind="optional_float", lo=0.0, hi=1.0,
        suggested=0.25, help="The falloff must also be at least this fraction of the peak speed.")

    def validate(self) -> list[str]:
        errors = super().validate()
        if self.min_falloff_abs is None and self.min_falloff_rel is None:
            errors.append("switch on at least one falloff threshold")
        if len(set(self.profile_booms)) < 3:
            errors.append("the profile needs at least three booms")
        extra = sorted(set(self.required_booms) - set(self.profile_booms))
        if extra:
            errors.append(f"required booms {extra} aren't profile booms")
        return errors


def _pick(mask: np.ndarray, key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row, the column of the largest `key` where `mask` (ties: the lowest
    column, i.e. the lower boom) and whether there was one.
    """
    return np.argmax(np.where(mask, key, -np.inf), axis=1), mask.any(axis=1)


@register("nose", label="Peak (local maximum with falloff)", params=NoseParams)
def nose(profiles: Profiles, params: NoseParams) -> pd.DataFrame:
    """The peak is a strict local maximum of the profile - faster than the valid
    booms directly below and above it, so never the top boom - reaching the
    minimum peak speed, with the falloff to the slowest boom above it passing
    the absolute and/or relative threshold. The fastest such boom is the peak.
    Slots missing any required boom are skipped.
    """
    p = profiles.subset(params.profile_booms)
    ws, z = p.ws, p.heights
    n, m = ws.shape
    valid = np.isfinite(ws)
    has_data = valid[:, [p.column(b) for b in params.required_booms]].all(axis=1)

    # the slowest valid boom strictly above each column
    u_min = np.full((n, m), np.nan)
    z_min = np.full((n, m), np.nan)
    run_u, run_z = np.full(n, np.inf), np.full(n, np.nan)
    for j in range(m - 1, -1, -1):
        u_min[:, j] = np.where(np.isfinite(run_u), run_u, np.nan)
        z_min[:, j] = run_z
        slower = valid[:, j] & (ws[:, j] < run_u)
        run_u = np.where(slower, ws[:, j], run_u)
        run_z = np.where(slower, z[j], run_z)

    candidate = np.zeros((n, m), bool)
    mid, below, above = ws[:, 1:-1], ws[:, :-2], ws[:, 2:]
    candidate[:, 1:-1] = valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:] & (mid > below) & (mid > above)
    candidate &= has_data[:, None]

    falloff = ws - u_min
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = falloff / ws
    fast = candidate & (ws >= params.min_nose_speed)
    strong = np.ones_like(fast)
    if params.min_falloff_abs is not None:
        strong &= falloff >= params.min_falloff_abs
    if params.min_falloff_rel is not None:
        strong &= rel >= params.min_falloff_rel
    qualifies = fast & strong

    jet_col, is_jet = _pick(qualifies, ws)
    weak_col, has_fast = _pick(fast, falloff)
    slow_col, has_candidate = _pick(candidate, ws)

    status = np.full(n, NO_DATA, dtype=object)
    status[has_data] = "no_peak"
    status[has_candidate] = "too_slow"
    status[has_fast] = "weak_falloff"
    status[is_jet] = JET
    col = np.select([is_jet, has_fast, has_candidate], [jet_col, weak_col, slow_col], default=-1)

    rows, take, has = np.arange(n), np.clip(col, 0, None), col >= 0

    def at_nose(a):
        return np.where(has, a[rows, take], np.nan)
    return result_frame(p, status, NOSE_STATUSES, col, np.where(has, z[take], np.nan), at_nose(ws),
                        at_nose(z_min), at_nose(u_min))
