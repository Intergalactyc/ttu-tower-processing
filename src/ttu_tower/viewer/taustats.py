"""tau over the whole run: each (slot, boom, variant)'s heat and momentum
detections and the selected tau, as categories (the ladder rungs, then the
statuses without a tau) for distributions, statistics, the heat-vs-momentum
joint distribution, and a what-if of the selection rules over every slot.

The what-if reruns secondary's own `select` on each distinct pair of
detections: tau takes only the rung values, so a year's selections reduce to
a few hundred distinct cases.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ttu_tower.secondary.detect import STATUSES, Detection
from ttu_tower.secondary.select import select

RUNGS = (9.375, 18.75, 37.5, 75.0, 150.0, 300.0, 600.0, 1200.0)
VARIANTS = ("mrd", "mrd_unexcised")
SOURCES = ("heat", "momentum", "fallback", "none")
WHICH = {"heat": "heat τ", "momentum": "momentum τ", "selected": "selected τ"}
# the categories past the rungs, by quantity
FAMILY_EXTRA = ("unresolved", "weak", "no_data")
SELECTED_EXTRA = ("none",)
_ACTIVE = ("found", "capped")


def labels(which: str) -> list[str]:
    rungs = [f"{r:g}" for r in RUNGS]
    return rungs + list(SELECTED_EXTRA if which == "selected" else FAMILY_EXTRA)


@dataclass
class TauData:
    """One row per (slot, boom, variant), the variants in VARIANTS."""
    slot: np.ndarray
    boom: np.ndarray
    variant: np.ndarray  # index into VARIANTS
    status: dict[str, np.ndarray]  # family -> index into STATUSES
    tau: dict[str, np.ndarray]  # family -> tau_s (NaN unless found/capped)
    tau_lb: dict[str, np.ndarray]  # family -> the lower bound of an unresolved tau
    sel_tau: np.ndarray
    sel_source: np.ndarray  # index into SOURCES
    sel_status: np.ndarray  # the source's own status, as stored


def load(run, index) -> TauData:
    """The run's tau and tau_selected, cached on the run."""
    return run.cache("taustats", 2).get("tau", lambda: _load(index))


def _codes(values, vocabulary) -> np.ndarray:
    lookup = {v: i for i, v in enumerate(vocabulary)}
    return np.array([lookup.get(v, -1) for v in values], dtype=np.int8)


def _load(index) -> TauData:
    tau = index.read("tau", columns=["slot", "boom", "variant", "cospectrum", "status", "tau_s", "tau_lb_s"])
    sel = index.read("tau_selected", columns=["slot", "boom", "variant", "tau_s", "source", "source_status"])
    return frame_data(tau, sel)


def frame_data(tau: pd.DataFrame, sel: pd.DataFrame) -> TauData:
    for df in (tau, sel):
        df["variant"] = df["variant"].astype(str)
    keys = ["slot", "boom", "variant"]
    tau = tau[tau["variant"].isin(VARIANTS)]
    wide = sel[sel["variant"].isin(VARIANTS)][keys + ["tau_s", "source", "source_status"]].rename(
        columns={"tau_s": "sel_tau"})
    for family in ("heat", "momentum"):
        part = tau[tau["cospectrum"].astype(str) == family][keys + ["status", "tau_s", "tau_lb_s"]]
        part = part.rename(columns={"status": f"{family}_status", "tau_s": f"{family}_tau",
                                    "tau_lb_s": f"{family}_lb"})
        wide = wide.merge(part, on=keys, how="left")
    wide = wide.sort_values(keys, kind="stable").reset_index(drop=True)
    status = {f: _codes(wide[f"{f}_status"].astype(object).fillna("no_data"), STATUSES) for f in ("heat", "momentum")}
    return TauData(
        slot=wide["slot"].to_numpy(np.int64), boom=wide["boom"].to_numpy(np.int8),
        variant=_codes(wide["variant"], VARIANTS), status=status,
        tau={f: wide[f"{f}_tau"].to_numpy(float) for f in ("heat", "momentum")},
        tau_lb={f: wide[f"{f}_lb"].to_numpy(float) for f in ("heat", "momentum")},
        sel_tau=wide["sel_tau"].to_numpy(float), sel_source=_codes(wide["source"].astype(str), SOURCES),
        sel_status=wide["source_status"].astype(str).to_numpy(object),
    )


def _rung_index(tau: np.ndarray) -> np.ndarray:
    """Index into RUNGS of each tau; -1 where it isn't a rung (or NaN)."""
    rungs = np.asarray(RUNGS)
    i = np.clip(np.searchsorted(rungs, tau), 0, rungs.size - 1)
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(tau) & np.isclose(rungs[i], tau), i, -1)


def categories(data: TauData, which: str, sel_tau=None, sel_source=None) -> np.ndarray:
    """Each row's category: the rung of its tau (0-7), else the status past them
    (see `labels`). `sel_tau`/`sel_source` stand in for the stored selection.
    """
    n = len(RUNGS)
    if which == "selected":
        tau = data.sel_tau if sel_tau is None else sel_tau
        source = data.sel_source if sel_source is None else sel_source
        out = _rung_index(tau)
        return np.where(source == SOURCES.index("none"), n, np.where(out < 0, n, out)).astype(np.int16)
    status = data.status[which]
    active = np.isin(status, [STATUSES.index(s) for s in _ACTIVE])
    out = np.where(active, _rung_index(data.tau[which]), -1)
    for k, name in enumerate(FAMILY_EXTRA):
        out = np.where(status == STATUSES.index(name), n + k, out)
    return np.where(out < 0, n + FAMILY_EXTRA.index("no_data"), out).astype(np.int16)


def tau_values(cats: np.ndarray) -> np.ndarray:
    """The tau of each rung category; NaN for the others."""
    out = np.full(cats.size, np.nan)
    rung = cats < len(RUNGS)
    out[rung] = np.asarray(RUNGS)[cats[rung]]
    return out


def shares(cats: np.ndarray, which: str) -> np.ndarray:
    """% of rows in each category (all categories, `labels(which)` order)."""
    counts = np.bincount(cats, minlength=len(labels(which))).astype(float)
    return 100.0 * counts / max(counts.sum(), 1.0)


def summary(cats: np.ndarray, which: str, source: np.ndarray | None = None, status: np.ndarray | None = None) -> dict:
    """N; % by status (for the selected tau: by source); the median and
    quartiles of the tau values (rung categories only), on the rungs.
    """
    out = {"N": int(cats.size)}
    if which == "selected":
        for i, name in enumerate(SOURCES):
            out[f"% {name}"] = 100.0 * float(np.mean(source == i)) if cats.size else np.nan
    else:
        for i, name in enumerate(STATUSES):
            out[f"% {name}"] = 100.0 * float(np.mean(status == i)) if cats.size else np.nan
    taus = tau_values(cats)
    taus = taus[np.isfinite(taus)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        q = np.percentile(taus, [25, 50, 75], method="nearest") if taus.size else [np.nan] * 3
    out["median τ"], out["p25 τ"], out["p75 τ"] = (float(v) for v in (q[1], q[0], q[2]))
    return out


def joint(heat: np.ndarray, momentum: np.ndarray) -> np.ndarray:
    """Counts, heat categories (rows) × momentum categories (columns)."""
    n = len(labels("heat"))
    return np.bincount(heat.astype(np.int64) * n + momentum, minlength=n * n).reshape(n, n)


def agreement(heat: np.ndarray, momentum: np.ndarray) -> dict:
    """Where both detections have a tau: how often they agree, which is longer,
    and the median log2(heat tau / momentum tau) (in rungs).
    """
    both = (heat < len(RUNGS)) & (momentum < len(RUNGS))
    h, m = heat[both].astype(int), momentum[both].astype(int)
    n = int(both.sum())
    if n == 0:
        return {"N both": 0, "% equal": np.nan, "% heat longer": np.nan, "% momentum longer": np.nan,
                "median rungs h−m": np.nan}
    return {"N both": n, "% equal": 100.0 * float(np.mean(h == m)), "% heat longer": 100.0 * float(np.mean(h > m)),
            "% momentum longer": 100.0 * float(np.mean(h < m)), "median rungs h−m": float(np.median(h - m))}


def reselect(data: TauData, cfg_sel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(tau_s, source index, source status) of every row under `cfg_sel`,
    from secondary's own `select` on each distinct pair of detections.
    """
    def det(family, i):
        return Detection(STATUSES[data.status[family][i]], float(data.tau[family][i]),
                         float(data.tau_lb[family][i]), 0, np.nan, np.nan, None)

    cols = [data.status["heat"], data.tau["heat"], data.tau_lb["heat"],
            data.status["momentum"], data.tau["momentum"], data.tau_lb["momentum"]]
    key = pd.DataFrame({i: c for i, c in enumerate(cols)})
    group = key.groupby(list(range(6)), dropna=False, sort=False).ngroup().to_numpy()
    first = pd.Series(np.arange(group.size)).groupby(group).first().to_numpy()
    tau = np.empty(first.size)
    source = np.empty(first.size, dtype=np.int8)
    status = np.empty(first.size, dtype=object)
    for g, i in enumerate(first):
        t, s, st = select(det("heat", i), det("momentum", i), cfg_sel)
        tau[g], source[g], status[g] = t, SOURCES.index(s), st
    return tau[group], source[group], status[group]
