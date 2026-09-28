"""QC what-if: the run's `[qc]` parameters as editable fields, a config with
some of them changed (checked by the config loader's own rules), a second
Reprocessor that runs Stage B with it, and what changed against the run's
stored rows - slot coverage and means, and the focus slot's tau.
"""
import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.dataset as pa_dataset

from ttu_tower.config.load import ConfigError, config_from_dict, to_resolved_dict
from ttu_tower.flags import TESTS
from ttu_tower.primary.products import file_products
from ttu_tower.viewer import whatif
from ttu_tower.viewer.reprocess import Reprocessor
from ttu_tower.viewer.slotdata import SlotAcross, _plain

QUALITY_TESTS = tuple(name for name, spec in TESTS.items() if spec.kind == "quality")
_SECTION_TITLES = {"": "coverage and gaps", "bounds": "bounds", "despike": "despiking", "windows": "window tests",
                   "slow": "slow-sensor smoothing", "second_layer": "second layer (direction, bounce)"}


@dataclass(frozen=True)
class Field:
    path: tuple  # attribute names (and a dict key last, for per-variable settings)
    kind: str  # "float" | "int" | "pair" | "tests"
    section: str

    @property
    def label(self) -> str:
        return ".".join(str(p) for p in self.path[1:] if p) if self.section else ".".join(map(str, self.path))

    @property
    def name(self) -> str:
        return ".".join(map(str, self.path))


def fields(qc) -> list[Field]:
    """Every editable leaf of `[qc]`, grouped by section."""
    out = []

    def walk(obj, path, section):
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name)
            p = path + (f.name,)
            if dataclasses.is_dataclass(value):
                walk(value, p, f.name)
            elif isinstance(value, dict):
                out.extend(Field(p + (key,), "float", section) for key in value)
            elif f.name == "unusable_tests":
                out.append(Field(p, "tests", section))
            elif isinstance(value, tuple):
                out.append(Field(p, "pair", section))
            elif isinstance(value, bool):
                continue
            elif isinstance(value, int):
                out.append(Field(p, "int", section))
            else:
                out.append(Field(p, "float", section))

    walk(qc, (), "")
    return out


def section_title(section: str) -> str:
    return _SECTION_TITLES.get(section, section)


def get(qc, path: tuple):
    obj = qc
    for p in path:
        obj = obj[p] if isinstance(obj, dict) else getattr(obj, p)
    return obj


def _replace(obj, path: tuple, value):
    head, rest = path[0], path[1:]
    if isinstance(obj, dict):
        return {**obj, head: _replace(obj[head], rest, value) if rest else value}
    return dataclasses.replace(obj, **{head: _replace(getattr(obj, head), rest, value) if rest else value})


def with_values(cfg, values: dict[tuple, object]):
    """`cfg` with the given `[qc]` leaves (path -> value) replaced."""
    qc = cfg.qc
    for path, value in values.items():
        qc = _replace(qc, path, value)
    return dataclasses.replace(cfg, qc=qc)


def problems(cfg) -> list[str]:
    """What the config loader would reject in `cfg`."""
    try:
        config_from_dict(to_resolved_dict(cfg))
    except ConfigError as exc:
        return [str(exc)]
    return []


def changes(base_qc, qc) -> list[tuple[str, object, object]]:
    """(field, run's value, what-if value) for every field that differs."""
    return [(f.name, get(base_qc, f.path), get(qc, f.path)) for f in fields(base_qc)
            if get(base_qc, f.path) != get(qc, f.path)]


def reprocessor(base: Reprocessor, cfg) -> Reprocessor:
    """A Reprocessor for `cfg` with caches of its own. Stage A depends only on
    the bounds, so while they're the run's it reads the base's Stage A.
    """
    share = base if cfg.qc.bounds == base.cfg.qc.bounds else None
    return Reprocessor(base.run, cfg=cfg, stage_a_from=share)


# --- what changed ---------------------------------------------------------------------------

_COVERAGE = ("momentum", "heat", "ts", "t", "rh", "p")
_MEANS = ("ws", "wd", "w", "ts", "t", "rh", "p")


def _slot_rows(coverage: pd.DataFrame, means: pd.DataFrame, slots) -> pd.DataFrame:
    cov = coverage[(coverage["layer"].astype(str) == "usable") & coverage["variable"].astype(str).isin(_COVERAGE)]
    cov = pd.DataFrame({"slot": cov["slot"], "quantity": cov["variable"].astype(str) + " usable coverage",
                        "value": cov["fraction"]})
    mean = means[(means["stat"].astype(str) == "mean") & means["variable"].astype(str).isin(_MEANS)]
    mean = pd.DataFrame({"slot": mean["slot"], "quantity": mean["variable"].astype(str) + " mean",
                         "value": mean["value"]})
    rows = pd.concat([cov, mean], ignore_index=True)
    return rows[rows["slot"].isin(list(slots))]


def slot_comparison(index, rp: Reprocessor, boom: int, slots) -> pd.DataFrame:
    """Per slot: family coverage and slot means, as the run stored them and
    as the what-if QC gives them.
    """
    slots = list(slots)
    f = pa_dataset.field
    half_hours = sorted({k // 3 for k in slots})
    flt = (f("boom") == boom) & f("slot").isin(slots)
    stored = _slot_rows(index.read("coverage", booms=[boom], half_hours=half_hours, filter=flt),
                        index.read("means", booms=[boom], half_hours=half_hours, filter=flt), slots)
    fresh = [out for out in (rp.stage_b(boom, h)[0] for h in half_hours) if not out.is_placeholder]
    if fresh:
        new = _slot_rows(pd.concat([o.coverage for o in fresh], ignore_index=True),
                         pd.concat([o.means for o in fresh], ignore_index=True), slots)
    else:
        new = stored.iloc[:0]
    out = stored.merge(new, on=["slot", "quantity"], how="outer", suffixes=("_run", "_whatif"))
    out = out.rename(columns={"value_run": "run", "value_whatif": "what-if"})
    out["change"] = out["what-if"] - out["run"]
    order = {q: i for i, q in enumerate([f"{v} usable coverage" for v in _COVERAGE] + [f"{v} mean" for v in _MEANS])}
    out["_order"] = out["quantity"].map(order)
    return out.sort_values(["slot", "_order"]).drop(columns="_order").reset_index(drop=True)


def slot_tau(index, rp: Reprocessor, boom: int, k: int, cfg_secondary) -> pd.DataFrame:
    """Slot k's heat/momentum/selected tau and u*, σw as the run stored them,
    and as they come out of the what-if QC (primary's products recomputed, then
    secondary's detection and selection with the run's own parameters).
    """
    h = k // 3
    products = file_products(h, {hh: rp.stage_b(boom, hh)[0] for hh in range(h - 2, h + 3)}, boom, rp.cfg)

    def at_slot(name):
        df = products.get(name)
        return _plain(df[df["slot"] == k].copy()) if df is not None and "slot" in df.columns else None

    fresh = SlotAcross(slot=k, booms=(boom,), tables={n: t for n in ("coverage", "slot_boom", "ladder")
                                                      if (t := at_slot(n)) is not None}, mrd=at_slot("mrd"))
    new_tau, new_sel = whatif.rerun(fresh, cfg_secondary)
    f = pa_dataset.field
    flt = (f("boom") == boom) & (f("slot") == k)
    old_tau = _plain(index.read("tau", half_hours=[h], filter=flt))
    old_sel = _plain(index.read("tau_selected", half_hours=[h], filter=flt))
    old_ladder = _plain(index.read("ladder", booms=[boom], half_hours=[h], filter=flt))
    new_ladder = fresh.table("ladder")
    rows = []
    variants = sorted(set(new_sel["variant"]) | set(old_sel["variant"])) if not new_sel.empty or not old_sel.empty else []
    for variant in variants:
        if variant == "naive":
            continue
        row = {"variant": variant}
        for fam in ("heat", "momentum"):
            for side, tau in (("run", old_tau), ("what-if", new_tau)):
                r = tau[(tau["variant"] == variant) & (tau["cospectrum"] == fam)] if not tau.empty else tau
                row[f"{fam} ({side})"] = (whatif._describe(r["status"].iloc[0], r["tau_s"].iloc[0], r["tau_lb_s"].iloc[0])
                                          if not r.empty else "—")
        for side, sel, ladder in (("run", old_sel, old_ladder), ("what-if", new_sel, new_ladder)):
            r = sel[sel["variant"] == variant] if not sel.empty else sel
            tau_s = float(r["tau_s"].iloc[0]) if not r.empty else np.nan
            row[f"τ ({side})"] = tau_s
            row[f"source ({side})"] = f"{r['source'].iloc[0]} ({r['source_status'].iloc[0]})" if not r.empty else "—"
            ustar, sigma_w = whatif._ladder_values(ladder, boom, variant, tau_s)
            row[f"u* ({side})"], row[f"σw ({side})"] = ustar, sigma_w
        rows.append(row)
    return pd.DataFrame(rows)
