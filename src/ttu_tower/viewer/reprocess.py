"""Recomputing the 50-Hz series behind a slot from the raw files, with the
pipeline's own Stage A/B code and the run's config: "as measured" (units,
tilt and rotation only - nothing removed) or "QC applied" (the despiked,
gap-filled series with the final masks, and what each test removed).
"""
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ttu_tower.constants import SAMPLE_HZ, STAGE_B_MARGIN_S
from ttu_tower.flags import TESTS, FlagStore
from ttu_tower.io.load import BadFileError, load_boom
from ttu_tower.math.polar import rotate_streamwise, streamwise_angle, vector_to_bearing
from ttu_tower.primary.products import assemble_samples, half_hours_overlapping
from ttu_tower.primary.slow import vpts as compute_vpts
from ttu_tower.primary.stage_a import StageA, couple_triplet, rotate_to_earth, stage_a, tilt_correct, to_si
from ttu_tower.primary.stage_b import StageBOut, build_span, stage_b
from ttu_tower.timegrid import SAMPLES_PER_SLOT
from ttu_tower.viewer.cache import LRUCache
from ttu_tower.viewer.fragments import alias_bounds_variables

_MARGIN = STAGE_B_MARGIN_S * SAMPLE_HZ
_MAD_TO_SIGMA = 0.6745
SONIC = ("ue", "vn", "w")
DESPIKED = ("ue", "vn", "w", "ts", "t", "rh", "p")
SLOW = ("t", "rh", "p")
MODES = ("as_measured", "qc", "unexcised")


class RawDataMissing(Exception):
    """An accepted raw file is no longer where the run's file table says."""


@dataclass
class HFWindow:
    """50-Hz series over samples [g0, g0 + n) of one boom. Arrays are NaN (or
    False) wherever the window reaches a half-hour with no usable file.
    """

    boom: int
    g0: int
    n: int
    mode: str
    frame: str = "earth"
    series: dict[str, np.ndarray] = field(default_factory=dict)
    masks: dict[str, np.ndarray] = field(default_factory=dict)  # per variable: finally usable
    family_masks: dict[str, np.ndarray] = field(default_factory=dict)  # momentum / heat / ts
    filled: dict[str, np.ndarray] = field(default_factory=dict)  # interpolated by the <= 1-s gap fill
    removed: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)  # test -> variable -> mask
    slow_smoothed: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)  # (x, usable)
    despike_band: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)  # (lo, hi)
    missing_half_hours: list[int] = field(default_factory=list)

    @property
    def t_rel(self) -> np.ndarray:
        """Seconds from the window's first sample."""
        return np.arange(self.n) / SAMPLE_HZ


def slot_window(k: int, before_min: float = 10.0, after_min: float = 20.0) -> tuple[int, int]:
    """(g0, n) of a window from `before_min` before slot k's start to `after_min` after it."""
    g0 = SAMPLES_PER_SLOT * k - round(before_min * 60 * SAMPLE_HZ)
    return g0, round((before_min + after_min) * 60 * SAMPLE_HZ)


class Reprocessor:
    """Stage A and Stage B outputs per (boom, half-hour), cached, for one run."""

    def __init__(self, run, stage_a_cache: int = 64, stage_b_cache: int = 24):
        if run.cfg is None:
            raise ValueError("this run's config is unavailable, so it can't be reprocessed")
        self.run = run
        self.cfg = run.cfg
        self._A = LRUCache(stage_a_cache)
        self._B = LRUCache(stage_b_cache)
        self._tmp = tempfile.TemporaryDirectory(prefix="ttu_view_")

    def close(self) -> None:
        self._tmp.cleanup()

    # --- the cached pipeline steps ------------------------------------------------------

    def raw(self, boom: int, h: int) -> dict[str, np.ndarray] | None:
        """One boom's raw columns of half-hour h (None: no accepted or usable file)."""
        path = self.run.accepted.get(h)
        if path is None:
            return None
        if not Path(path).is_file():
            raise RawDataMissing(f"{path} is missing (was the raw data moved?)")
        try:
            return load_boom(path, boom, self._tmp.name)
        except BadFileError:
            return None

    def stage_a(self, boom: int, h: int) -> StageA | None:
        def compute():
            raw = self.raw(boom, h)
            return None if raw is None else stage_a(raw, boom, self.cfg.qc)
        return self._A.get((boom, h), compute)

    def stage_b(self, boom: int, h: int) -> tuple[StageBOut, dict]:
        """Half-hour h's Stage B output and its trace, exactly as primary computed it."""
        def compute():
            A = {hh: self.stage_a(boom, hh) for hh in (h - 1, h, h + 1)}
            trace = {}
            out = stage_b(build_span(A, h, _MARGIN), h, boom, self.cfg, trace=trace)
            return out, trace
        return self._B.get((boom, h), compute)

    def cached_stage_b(self, boom: int, h: int) -> bool:
        return (boom, h) in self._B

    # --- windows ---------------------------------------------------------------------------

    def as_measured(self, boom: int, g0: int, n: int, frame: str = "earth") -> HFWindow:
        """Units converted, tilt corrected and rotated (frame "earth": ue/vn/w),
        or units only (frame "sonic": the sonic's own u/v/w). Nothing removed.
        """
        per_hh: dict[str, dict[int, np.ndarray | None]] = {}
        missing = []
        for h in half_hours_overlapping(g0, n):
            raw = self.raw(boom, h)
            if raw is None:
                missing.append(h)
                continue
            si = to_si(raw)
            if frame == "sonic":
                cols = {"u": si["u"], "v": si["v"], "w": si["w"]}
            else:
                u, v, w = couple_triplet(si["u"], si["v"], si["w"])
                u, v, w = tilt_correct(u, v, w, boom)
                ue, vn = rotate_to_earth(u, v)
                cols = {"ue": ue, "vn": vn, "w": w}
            cols.update(ts=si["ts"], vpts=compute_vpts(si["ts"], boom), t=si["t"], rh=si["rh"], p=si["p"])
            for var, x in cols.items():
                per_hh.setdefault(var, {})[h] = x
        series = {var: assemble_samples(xs, g0, n, is_mask=False) for var, xs in per_hh.items()}
        if not series:
            names = ("u", "v", "w") if frame == "sonic" else SONIC
            series = {var: np.full(n, np.nan) for var in names + ("ts", "vpts") + SLOW}
        masks = {var: np.isfinite(x) for var, x in series.items()}
        return HFWindow(boom=boom, g0=g0, n=n, mode="as_measured", frame=frame, series=series, masks=masks,
                        missing_half_hours=missing)

    def qc_applied(self, boom: int, g0: int, n: int, variant: str = "mrd") -> HFWindow:
        """The series primary computed its statistics from: for "mrd" the
        despiked, <= 1-s-filled series with the final masks; for
        "mrd_unexcised" the within-run-filled counterfactual.
        """
        unexcised = variant == "mrd_unexcised"
        half_hours = half_hours_overlapping(g0, n)
        B = {h: self.stage_b(boom, h) for h in half_hours}
        missing = [h for h, (out, _) in B.items() if out.is_placeholder]
        live = {h: (out, tr) for h, (out, tr) in B.items() if not out.is_placeholder}

        def stitch(get, is_mask=False):
            return assemble_samples({h: get(out, tr) for h, (out, tr) in live.items()}, g0, n, is_mask=is_mask)

        win = HFWindow(boom=boom, g0=g0, n=n, mode="unexcised" if unexcised else "qc", missing_half_hours=missing)
        series_attr = "unexcised_series" if unexcised else "final_series"
        masks_attr = "unexcised_masks" if unexcised else "final_masks"
        for var in SONIC + ("ts", "vpts"):
            win.series[var] = stitch(lambda o, t, v=var: getattr(o, series_attr)[v])
            if unexcised:
                win.masks[var] = np.isfinite(win.series[var])
            else:
                win.masks[var] = stitch(lambda o, t, v=var: t["final_masks_var"][v], is_mask=True)
        for fam in ("momentum", "heat", "ts"):
            win.family_masks[fam] = stitch(lambda o, t, f=fam: getattr(o, masks_attr)[f], is_mask=True)

        for var in SLOW:
            x = stitch(lambda o, t, v=var: t["slow_smoothed"][v][0])
            ok = stitch(lambda o, t, v=var: t["slow_smoothed"][v][1], is_mask=True)
            win.slow_smoothed[var] = (x, ok)
            win.series[var] = np.where(ok, x, np.nan)
            win.masks[var] = ok

        for var in DESPIKED:
            win.filled[var] = stitch(lambda o, t, v=var: t["filled_mask"][v], is_mask=True)
            m = stitch(lambda o, t, v=var: t["despike_ref"][v][0].astype(np.float64))
            mad = stitch(lambda o, t, v=var: t["despike_ref"][v][1].astype(np.float64))
            half = self.cfg.qc.despike.z_threshold[var] * mad / _MAD_TO_SIGMA
            win.despike_band[var] = (m - half, m + half)

        flags = [out.flags for out, _ in live.values() if not out.flags.empty]
        win.removed = removed_by_test(pd.concat(flags, ignore_index=True) if flags else pd.DataFrame(), boom, g0, n)
        coupled = {var: stitch(lambda o, t, v=var: t["coupled"][v], is_mask=True) for var in SONIC}
        coupled = {var: m for var, m in coupled.items() if m.any()}
        if coupled:
            win.removed["coupled"] = coupled
        return win


def removed_by_test(flags: pd.DataFrame, boom: int, g0: int, n: int) -> dict[str, dict[str, np.ndarray]]:
    """test -> variable -> boolean mask over [g0, g0 + n), from flag intervals
    (boom-level tests folded into ue/vn/w/ts; vpts shares ts's flags).
    """
    store = FlagStore(alias_bounds_variables(flags) if not flags.empty else flags)
    removed = {}
    for test, spec in TESTS.items():
        per_var = {}
        for var in spec.variables or SONIC + ("ts",):
            mask = store.mask([test], boom, var, g0, n)
            if mask.any():
                per_var[var] = mask
        if "ts" in per_var:
            per_var["vpts"] = per_var["ts"]
        if per_var:
            removed[test] = per_var
    return removed


def to_streamwise(win: HFWindow, mean_by_slot: dict | None = None) -> tuple[HFWindow, dict[int, float]]:
    """`win` with ue/vn rotated, slot by slot, into each 10-min slot's own mean
    wind: u along it and v 90° to its left, so v averages to zero over the
    slot. A slot's mean wind comes from `mean_by_slot` (k -> (ue, vn), e.g.
    the run's stored slot means), or else from the window's own usable samples
    in it (all its finite samples, if none is usable). Returns the rotated window and each slot's wind FROM-bearing (deg).
    """
    ue, vn = win.series["ue"], win.series["vn"]
    usable = win.masks.get("ue", np.isfinite(ue)) & win.masks.get("vn", np.isfinite(vn))
    u, v = np.full(win.n, np.nan), np.full(win.n, np.nan)
    bearings = {}
    k0 = win.g0 // SAMPLES_PER_SLOT
    k1 = (win.g0 + win.n - 1) // SAMPLES_PER_SLOT
    for k in range(k0, k1 + 1):
        lo = max(k * SAMPLES_PER_SLOT - win.g0, 0)
        hi = min((k + 1) * SAMPLES_PER_SLOT - win.g0, win.n)
        mean = (mean_by_slot or {}).get(k)
        if mean is None or not np.all(np.isfinite(mean)):
            finite = np.isfinite(ue[lo:hi]) & np.isfinite(vn[lo:hi])
            ok = finite & usable[lo:hi]
            if not ok.any():
                ok = finite  # nothing usable: still rotate what is drawn, by its own mean
            if not ok.any():
                continue
            mean = (ue[lo:hi][ok].mean(), vn[lo:hi][ok].mean())
        phi = streamwise_angle(*mean)
        u[lo:hi], v[lo:hi] = rotate_streamwise(ue[lo:hi], vn[lo:hi], phi)
        bearings[k] = float(vector_to_bearing(*mean)[1])

    def swap(d: dict, combine) -> dict:
        out = {key: val for key, val in d.items() if key not in ("ue", "vn")}
        if "ue" in d or "vn" in d:
            pair = [d[key] for key in ("ue", "vn") if key in d]
            out["u"] = out["v"] = combine(pair)
        return out

    def union(masks):
        return np.logical_or.reduce(masks)

    def both(masks):
        return np.logical_and.reduce(masks)

    series = {key: val for key, val in win.series.items() if key not in ("ue", "vn")}
    series.update(u=u, v=v)
    rotated = HFWindow(
        boom=win.boom, g0=win.g0, n=win.n, mode=win.mode, frame="streamwise", series=series,
        masks=swap(win.masks, both), family_masks=win.family_masks, filled=swap(win.filled, union),
        removed={test: swap(per_var, union) for test, per_var in win.removed.items()},
        slow_smoothed=win.slow_smoothed,
        despike_band={key: val for key, val in win.despike_band.items() if key not in ("ue", "vn")},
        missing_half_hours=win.missing_half_hours,
    )
    return rotated, bearings
