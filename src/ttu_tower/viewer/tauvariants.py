"""τ variants the viewer adds to the run's own (mrd, naive, mrd_unexcised):
a fixed τ at any ladder rung, and named τ schemes - a selection rule (the
run's, or a what-if of detection and selection settings) with optional
overrides by stability class, classified either globally (a stability scheme,
per slot) or locally (each boom's own ζ and heat flux).

A variant is computed with the pipeline's own code - secondary's selected
statistics and derived quantities, tertiary's filtering and profile fits - on
the τ it picks for each (slot, boom), and kept in the ttu-tower home as
fragments in the run's own schema, which `FragmentIndex.read` serves
alongside the run's.
"""
from __future__ import annotations

import concurrent.futures
from concurrent.futures.process import BrokenProcessPool
import hashlib
import json
import multiprocessing
import os
import shutil
import threading
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pa_dataset

from ttu_tower import schema
from ttu_tower.flags import FlagStore
from ttu_tower.io import store as io_store
from ttu_tower.io.runs import locate_home
from ttu_tower.secondary.derived import derived, selected_stats
from ttu_tower.secondary.detect import SMOOTHING
from ttu_tower.tertiary import filtering, gradients, profiles
from ttu_tower.viewer import stability as criteria
from ttu_tower.viewer import taustats, whatif

RUNGS = taustats.RUNGS
STORED = ("mrd", "naive", "mrd_unexcised")
FIXED_PREFIX, SCHEME_PREFIX = "fixed_", "scheme:"
SECONDARY_TABLES = ("boom_stats", "boom_labels", "tau_selected")
TERTIARY_TABLES = ("boom_final", "boom_labels_final", "tau_final", "filter_log", "profile")
TABLES = SECONDARY_TABLES + TERTIARY_TABLES
LOCAL_NAMES = ("unstable", "neutral", "stable")
LOCAL_VARIABLES = {"zeta": "ζ, by the sign of the heat flux", "rig": "Ri_g, the gradient Richardson number"}
_RETIRED_LOCAL = {"rib:layer": "rig", "rib:log": "rig"}  # saved schemes from before Ri_g
DETECTION_FIELDS = ("peak_significance_se", "min_scale_s", "min_tau_s", "max_tau_s")
SELECTION_FIELDS = ("rule", "priority", "fallback_tau_s")
_VERSION = 2  # 2: the gradient flux products (rif, phi_m, ...) in boom_final


# --- names ---------------------------------------------------------------------------------------

def fixed_key(rung: float) -> str:
    return f"{FIXED_PREFIX}{rung:g}"


ALL_FIXED = tuple(fixed_key(r) for r in RUNGS)
FIXED = tuple(k for k in ALL_FIXED if k != fixed_key(600.0))  # in menus: 600 s is the run's own naive


def scheme_key(name: str) -> str:
    return f"{SCHEME_PREFIX}{name}"


def is_virtual(variant: str | None) -> bool:
    return bool(variant) and variant.startswith((FIXED_PREFIX, SCHEME_PREFIX))


def fixed_rung(variant: str) -> float | None:
    return float(variant[len(FIXED_PREFIX):]) if variant.startswith(FIXED_PREFIX) else None


def rung_text(rung: float) -> str:
    return f"{rung:g} s" + (f" ({rung / 60:g} min)" if rung >= 300 else "")


def label(variant: str) -> str:
    if variant.startswith(FIXED_PREFIX):
        return f"τ = {rung_text(fixed_rung(variant))}"
    if variant.startswith(SCHEME_PREFIX):
        return f"τ scheme “{variant[len(SCHEME_PREFIX):]}”"
    return variant


def tip(variant: str) -> str:
    if variant.startswith(FIXED_PREFIX):
        return (f"Every slot at the {fixed_rung(variant):g}-s rung of mrd (computed by the viewer from the ladder, "
                "with the pipeline's own derivation and filtering)")
    if variant.startswith(SCHEME_PREFIX):
        return "A τ scheme of Controls → τ (computed by the viewer from the ladder)"
    return ""


def menu_variants(scheme_names) -> list[str]:
    """The viewer's variants, in menu order: the fixed rungs, then the schemes."""
    return list(FIXED) + [scheme_key(n) for n in sorted(scheme_names, key=str.lower)]


_BASES: dict[str, str] = {}  # scheme key -> its data ("mrd" or "mrd_unexcised"), as last defined


def basis(variant: str) -> str:
    """The run variant whose spectra and ladder a variant is computed from
    (naive's and every fixed rung's is mrd).
    """
    if variant == "naive" or variant.startswith(FIXED_PREFIX):
        return "mrd"
    if variant.startswith(SCHEME_PREFIX):
        return _BASES.get(variant, "mrd")
    return variant


def add_to(catalog, keys) -> None:
    """Offer `keys` (viewer variants) on every quantity that has the run's τ variants."""
    from dataclasses import replace as _replace
    for q in list(catalog):
        if "mrd" in q.variants and q.table in ("boom_final", "profile", "boom_labels_final", "tau_final"):
            own = tuple(v for v in q.variants if not is_virtual(v))
            catalog.quantities[q.key] = _replace(q, variants=own + tuple(keys))


def choice_label(choice: str) -> str:
    if choice == "base":
        return "the base selection"
    if choice == "run":
        return "the run's selection"
    return f"τ = {rung_text(float(choice))}"


def choices() -> list[str]:
    return ["base", "run"] + [f"{r:g}" for r in RUNGS]


# --- schemes ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TauScheme:
    data: str = "mrd"  # "mrd" (excised) or "mrd_unexcised"
    base: str = "run"  # "run": the run's own selection; "whatif": these settings
    detection: tuple = ()  # ((field, value), ...) replacing the run's detection; () = the run's
    kernel: tuple[float, ...] = SMOOTHING
    selection: tuple = ()  # ((field, value), ...) replacing the run's selection; priority as "heat,momentum"
    classifier: str = "none"  # "none", "global" (a stability scheme) or "local" (each boom's ζ)
    stability: criteria.StabilitySpec | None = None  # global: a copy of the scheme it classifies by
    stability_source: str = ""  # what that copy was taken from
    local_variable: str = "zeta"  # what classifies locally: a key of LOCAL_VARIABLES
    local_mode: str = "percent"  # "zeta0": |x| ≤ zeta0 is neutral; "percent": each boom's percent closest to 0
    zeta0: float = 0.05
    percent: float = 20.0
    zeta_variant: str = "mrd"  # whose ζ and heat flux classify locally (Ri has no variant)
    choices: tuple[str, ...] = ()  # per class: "base", "run" or a rung
    otherwise: str = "base"  # unclassified rows (every row without a classifier)

    def class_names(self) -> tuple[str, ...]:
        return criteria.NAMES if self.classifier == "global" else LOCAL_NAMES if self.classifier == "local" else ()

    def errors(self) -> list[str]:
        out = []
        if self.data not in ("mrd", "mrd_unexcised"):
            out.append(f"no data {self.data!r}")
        valid = set(choices())
        if any(c not in valid for c in (*self.choices, self.otherwise)):
            out.append("a class's τ isn't one of the choices")
        if self.classifier == "global":
            if self.stability is None:
                out.append("choose the stability scheme to classify by")
            else:
                out += self.stability.errors()
                if self.stability.variable == "zeta" and self.stability.variant.startswith(SCHEME_PREFIX):
                    out.append("a τ scheme can't classify by another τ scheme's ζ")
        elif self.classifier == "local":
            if self.local_variable not in LOCAL_VARIABLES:
                out.append(f"no local variable {self.local_variable!r}")
            if self.local_variable == "zeta" and self.zeta_variant.startswith(SCHEME_PREFIX):
                out.append("a τ scheme can't classify by another τ scheme's ζ")
            if self.local_mode == "zeta0" and not self.zeta0 > 0:
                out.append("the neutral threshold must be > 0")
            if self.local_mode == "percent" and not 0 < self.percent < 100:
                out.append("the neutral share must be between 0 and 100 %")
        elif self.classifier != "none":
            out.append(f"no classifier {self.classifier!r}")
        if self.classifier != "none" and len(self.choices) != len(self.class_names()):
            out.append("one τ choice per class is needed")
        return out

    def depends_on(self) -> list[str]:
        """The viewer variants whose ζ this scheme classifies by."""
        if self.classifier == "global" and self.stability is not None and self.stability.variable == "zeta":
            v = self.stability.variant
        elif self.classifier == "local" and self.local_variable == "zeta":
            v = self.zeta_variant
        else:
            v = None
        return [v] if v is not None and is_virtual(v) else []

    def local_symbol(self) -> str:
        return {"zeta": "ζ", "rig": "Ri_g"}.get(self.local_variable, self.local_variable)

    def secondary_config(self, cfg_secondary):
        """The run's secondary config with this scheme's detection and selection settings."""
        sel = dict(self.selection)
        if "priority" in sel:
            sel["priority"] = tuple(str(sel["priority"]).split(","))
        return whatif.edited(cfg_secondary, detection=dict(self.detection), selection=sel)

    def detection_config(self, cfg_secondary):
        return self.secondary_config(cfg_secondary).detection

    def selection_config(self, cfg_secondary):
        return self.secondary_config(cfg_secondary).selection

    def describe(self) -> str:
        parts = ["unexcised" if self.data == "mrd_unexcised" else "excised"]
        parts.append("base: the run's selection" if self.base == "run" else "base: a what-if selection")
        if self.classifier == "none":
            if self.otherwise != "base":
                parts.append(f"always {choice_label(self.otherwise)}")
        else:
            by = (self.stability.describe() if self.classifier == "global" and self.stability is not None
                  else f"local {self.local_symbol()}" + (f" ({self.zeta_variant})" if self.local_variable == "zeta" else "")
                  + (f", |{self.local_symbol()}| ≤ {self.zeta0:g} neutral" if self.local_mode == "zeta0"
                     else f", {self.percent:g}% neutral per boom"))
            parts.append(f"by {by}: " + "; ".join(f"{n} → {choice_label(c)}"
                                                  for n, c in zip(self.class_names(), self.choices)))
            parts.append(f"otherwise {choice_label(self.otherwise)}")
        return " · ".join(parts)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["detection"] = dict(self.detection)
        d["selection"] = dict(self.selection)
        d["kernel"] = list(self.kernel)
        d["choices"] = list(self.choices)
        d["stability"] = self.stability.to_dict() if self.stability is not None else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TauScheme:
        d = dict(d)
        d["detection"] = tuple(sorted((k, float(v)) for k, v in (d.get("detection") or {}).items()))
        d["selection"] = tuple(sorted((k, float(v) if k == "fallback_tau_s" else str(v))
                                      for k, v in (d.get("selection") or {}).items()))
        d["kernel"] = tuple(float(w) for w in d.get("kernel", SMOOTHING))
        d["choices"] = tuple(str(c) for c in d.get("choices", ()))
        d["stability"] = criteria.StabilitySpec.from_dict(d["stability"]) if d.get("stability") else None
        d["local_variable"] = _RETIRED_LOCAL.get(d.get("local_variable"), d.get("local_variable", "zeta"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def fixed_scheme(rung: float) -> TauScheme:
    return TauScheme(otherwise=f"{rung:g}")


def schemes_path() -> Path | None:
    home = locate_home()
    return home / "viewer" / "tau_schemes.json" if home is not None else None


@dataclass
class SchemeStore:
    """name -> TauScheme. Every change is written through; `path` None keeps them in memory only."""
    path: Path | None = None
    items: dict[str, TauScheme] = field(default_factory=dict)

    @classmethod
    def open(cls, path: Path | None = None, load: bool = True) -> SchemeStore:
        store = cls(Path(path) if path is not None else (schemes_path() if load else None))
        if store.path is not None and store.path.exists():
            raw = json.loads(store.path.read_text(encoding="utf-8"))
            store.items = {name: TauScheme.from_dict(s) for name, s in raw.get("schemes", {}).items()}
        return store

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"version": 1, "schemes": {n: s.to_dict() for n, s in self.items.items()}},
                                  indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    def put(self, name: str, scheme: TauScheme) -> None:
        self.items[name] = scheme
        self.save()

    def remove(self, name: str) -> None:
        self.items.pop(name, None)
        self.save()

    def name_of(self, scheme: TauScheme) -> str | None:
        return next((n for n, s in sorted(self.items.items()) if s == scheme), None)


# --- the variants of one run ---------------------------------------------------------------------

class StaticVariants:
    """Already computed variants (key -> directory), for a worker's reads."""

    def __init__(self, dirs: dict[str, Path]):
        self.dirs = {k: Path(v) for k, v in dirs.items()}

    def ready(self) -> dict[str, Path]:
        return dict(self.dirs)

    def ensure(self, key: str) -> Path | None:
        return self.dirs.get(key)

    def known(self) -> list[str]:
        return list(self.dirs)


class VariantStore:
    """The viewer's τ variants of one run: their definitions (the fixed rungs
    and the τ schemes), and where each one computed is kept. `ensure` computes
    a variant once (concurrent requests wait for the one computation).
    `workers`: processes, one boom each (0 computes in this process).
    """

    def __init__(self, run, index, root: Path | None = None, workers: int | None = None):
        self.run, self.index = run, index
        home = locate_home()
        self.root = Path(root) if root is not None else (home / "viewer" / "tau_variants" if home else None)
        self.workers = workers
        self.schemes: dict[str, TauScheme] = {}
        self.progress = None  # (key, text) -> None, from whichever thread computes
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self._ready: dict[str, Path] = {}

    # definitions

    def definition(self, key: str) -> TauScheme | None:
        if key.startswith(FIXED_PREFIX):
            return fixed_scheme(fixed_rung(key))
        if key.startswith(SCHEME_PREFIX):
            return self.schemes.get(key[len(SCHEME_PREFIX):])
        return None

    def known(self) -> list[str]:
        return list(ALL_FIXED) + [scheme_key(n) for n in self.schemes]

    def set_schemes(self, schemes: dict[str, TauScheme]) -> list[str]:
        """The schemes now defined; returns the keys whose definition changed (their data goes stale)."""
        old, self.schemes = self.schemes, dict(schemes)
        changed = [scheme_key(n) for n in set(old) | set(schemes) if old.get(n) != schemes.get(n)]
        _BASES.update({scheme_key(n): s.data for n, s in schemes.items()})
        with self._guard:
            for key in changed:
                self._ready.pop(key, None)
        return changed

    def run_identity(self) -> dict:
        meta = self.run.meta
        return {s: {k: (meta.get(s) or {}).get(k) for k in ("created", "config_hash")}
                for s in ("secondary", "tertiary")}

    def digest(self, key: str) -> str:
        scheme = self.definition(key)
        text = json.dumps({"v": _VERSION, "scheme": scheme.to_dict(), "run": self.run_identity()}, sort_keys=True,
                          default=str)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

    def directory(self, key: str) -> Path | None:
        if self.root is None or self.definition(key) is None:
            return None
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in key)
        return self.root / self.run.tag / f"{safe}-{self.digest(key)}"

    def ready(self) -> dict[str, Path]:
        """key -> directory of every defined variant already computed."""
        with self._guard:
            out = dict(self._ready)
        for key in self.known():
            if key not in out:
                d = self.directory(key)
                if d is not None and (d / "done.json").is_file():
                    out[key] = d
                    with self._guard:
                        self._ready[key] = d
        return out

    def is_ready(self, key: str) -> bool:
        return key in self.ready()

    def summary(self, key: str) -> dict | None:
        d = self.ready().get(key)
        return json.loads((d / "done.json").read_text(encoding="utf-8")) if d is not None else None

    def ensure(self, key: str) -> Path | None:
        """The variant's directory, computing it first if needed (None if it can't be)."""
        d = self.ready().get(key)
        if d is not None:
            return d
        scheme = self.definition(key)
        if scheme is None or self.root is None:
            return None
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            d = self.ready().get(key)
            if d is not None:
                return d
            for dep in scheme.depends_on():
                self.ensure(dep)
            d = self.directory(key)
            compute(self.run, self.index, key, scheme, d, workers=self.workers, progress=self._report)
            for old in d.parent.glob(f"{d.name.rsplit('-', 1)[0]}-*"):
                if old != d and old.is_dir():
                    shutil.rmtree(old, ignore_errors=True)
            with self._guard:
                self._ready[key] = d
            return d

    def _report(self, key: str, text: str) -> None:
        if self.progress is not None:
            self.progress(key, text)

    def disk_usage(self) -> int:
        base = self.root / self.run.tag if self.root is not None else None
        if base is None or not base.is_dir():
            return 0
        return sum(p.stat().st_size for p in base.rglob("*") if p.is_file())

    def clear(self) -> None:
        """Forget every computed variant of this run."""
        with self._guard:
            self._ready.clear()
        if self.root is not None:
            shutil.rmtree(self.root / self.run.tag, ignore_errors=True)


# --- reading them --------------------------------------------------------------------------------

def requested(variants, flt) -> list[str]:
    """The viewer variants a read with filter `flt` wants: the ones it names,
    else (no variant in it) every one already computed.
    """
    text = "" if flt is None else str(flt)
    if "variant" not in text:
        return list(variants.ready())
    return [k for k in variants.known() if f'"{k}"' in text]


def paths(directory: Path, table: str, booms=None) -> list[Path]:
    table_dir = Path(directory) / table
    if not table_dir.is_dir():
        return []
    if table == "profile" or booms is None:
        return sorted(table_dir.glob("*.parquet"))
    return [p for b in booms if (p := table_dir / f"b{int(b):02d}.parquet").is_file()]


def rules(directory: Path, booms=None) -> pd.DataFrame:
    """slot, boom, rule: which class (or "otherwise") chose each row's τ."""
    return io_store.read_paths(paths(directory, "tau_rule", booms))


# --- computing one ---------------------------------------------------------------------------------

def compute(run, index, key: str, scheme: TauScheme, directory: Path, workers: int | None = None,
            progress=None) -> None:
    """Every table of variant `key` into `directory` (written whole, then renamed into place)."""
    cfg = run.cfg
    if cfg is None:
        raise RuntimeError("the run's config is unavailable, so τ variants can't be computed")
    problems = scheme.errors()
    if scheme.base == "whatif":
        problems += whatif.problems(scheme.secondary_config(cfg.secondary))
    if problems:
        raise ValueError("; ".join(problems))
    started = time.time()
    say = (lambda text: progress(key, text)) if progress is not None else (lambda text: None)
    tmp = directory.with_name(directory.name + f".tmp{os.getpid()}-{threading.get_ident()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    global_classes = None
    if scheme.classifier == "global":
        series = criteria.values(index, scheme.stability)
        names = criteria.classify(series, scheme.stability.classes())
        codes = pd.Categorical(names.to_numpy(dtype=object), categories=list(criteria.NAMES)).codes
        global_classes = pd.Series(codes.astype(np.int16), index=series.index)
    deps = {d: index.variants.ensure(d) for d in scheme.depends_on()} if index.variants is not None else {}

    booms = list(run.booms)
    jobs = [dict(run_dir=str(run.run_dir), boom=b, key=key, scheme=scheme.to_dict(), cfg=cfg, out=str(tmp),
                 deps={k: str(v) for k, v in deps.items() if v is not None}, global_classes=global_classes)
            for b in booms]
    results = {}
    say(f"computing {label(key)}: 0 of {len(booms)} booms")
    n = len(booms) if workers is None else workers
    n = min(n, max((os.cpu_count() or 2) - 1, 1), len(booms))
    if n <= 0 or len(booms) == 1:
        for job in jobs:
            results[job["boom"]] = boom_part(job)
            say(f"computing {label(key)}: {len(results)} of {len(booms)} booms")
    else:
        context = multiprocessing.get_context("spawn")
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=n, mp_context=context) as pool:
                futures = {pool.submit(boom_part, job): job["boom"] for job in jobs}
                for future in concurrent.futures.as_completed(futures):
                    results[futures[future]] = future.result()
                    say(f"computing {label(key)}: {len(results)} of {len(booms)} booms")
        except BrokenProcessPool:  # workers couldn't start: here, one boom at a time
            for job in jobs:
                if job["boom"] not in results:
                    results[job["boom"]] = boom_part(job)
                    say(f"computing {label(key)}: {len(results)} of {len(booms)} booms")

    say(f"computing {label(key)}: gradients and profile fits")
    _gradients(run, index, key, tmp, cfg)
    _profile(run, index, key, tmp, cfg)
    summary = {"key": key, "scheme": scheme.to_dict(), "run": {"tag": run.tag}, "seconds": round(time.time() - started, 1),
               "booms": {str(b): r for b, r in sorted(results.items())}}
    (tmp / "done.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    shutil.rmtree(directory, ignore_errors=True)
    directory.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, directory)
    say(f"{label(key)} computed in {summary['seconds']:g} s")


def _obj(df: pd.DataFrame, columns) -> pd.DataFrame:
    for c in columns:
        if c in df.columns:
            df[c] = df[c].astype(object)
    return df


def _slot_starts(run_dir: Path) -> pd.Series:
    slots = pd.read_parquet(Path(run_dir) / "primary" / "slots.parquet", columns=["slot", "slot_start"])
    return slots.set_index("slot")["slot_start"]


def _write(out: Path, table: str, boom: int, df: pd.DataFrame, starts: pd.Series | None = None) -> None:
    if starts is not None:
        df = df.assign(slot_start=df["slot"].map(starts))
    df = df.sort_values(["slot"], kind="stable") if "slot" in df.columns else df
    if table == "tau_rule":
        io_store.write_fragment(out / table, f"b{boom:02d}", df.reset_index(drop=True))
        return
    io_store.write_fragment(out / table, f"b{boom:02d}" if boom >= 0 else "all",
                            schema.cast(table, df.reset_index(drop=True)))


def _selection_rows(index, boom: int, variant: str) -> pd.DataFrame:
    f = pa_dataset.field
    rows = index.read("tau_selected", columns=["slot", "tau_s", "source", "source_status"],
                      filter=(f("boom") == boom) & (f("variant") == variant))
    return _obj(rows, ("source", "source_status")).sort_values("slot", kind="stable").reset_index(drop=True)


def _whatif_rows(index, boom: int, scheme: TauScheme, cfg) -> pd.DataFrame:
    """The base selection of `boom` under the scheme's what-if settings."""
    f = pa_dataset.field
    tau = index.read("tau", columns=["slot", "boom", "variant", "cospectrum", "status", "tau_s", "tau_lb_s"],
                     filter=f("boom") == boom)
    sel = index.read("tau_selected", columns=["slot", "boom", "variant", "tau_s", "source", "source_status"],
                     filter=f("boom") == boom)
    data = taustats.frame_data(tau, sel)
    det, kernel = scheme.detection_config(cfg.secondary), tuple(scheme.kernel)
    if (det, kernel) != (cfg.secondary.detection, SMOOTHING):
        data = taustats.with_detections(data, {boom: taustats.redetect(index, data, boom, det, kernel)})
    t, s, st = taustats.reselect(data, scheme.selection_config(cfg.secondary))
    keep = data.variant == taustats.VARIANTS.index(scheme.data)
    return pd.DataFrame({"slot": data.slot[keep], "tau_s": t[keep],
                         "source": np.asarray(taustats.SOURCES, dtype=object)[s[keep]],
                         "source_status": st[keep]}).reset_index(drop=True)


def _local_classes(index, boom: int, scheme: TauScheme, slots: np.ndarray) -> tuple[np.ndarray, float]:
    """(class code per slot: 0 unstable, 1 neutral, 2 stable, -1 none; the |x| threshold). ζ splits the
    rest by the heat flux's sign (w'θv' > 0 unstable), Ri_g by its own (< 0 unstable).
    """
    f = pa_dataset.field
    if scheme.local_variable != "zeta":
        rows = index.read("boom_final", columns=["slot", "value"],
                          filter=(f("boom") == boom) & (f("variant") == "none")
                          & (f("variable") == scheme.local_variable) & f("stat").is_null())
        ri = rows.set_index("slot")["value"].reindex(slots).to_numpy(float)
        return _split(ri, -ri, scheme)
    rows = index.read("boom_final", columns=["slot", "variable", "value"],
                      filter=(f("boom") == boom) & (f("variant") == scheme.zeta_variant)
                      & (((f("variable") == "zeta") & f("stat").is_null())
                         | ((f("variable") == "wvpts") & (f("stat") == "cov"))))
    rows = _obj(rows, ("variable",))
    wide = rows.pivot_table(index="slot", columns="variable", values="value", aggfunc="first")
    zeta = wide["zeta"].reindex(slots).to_numpy(float) if "zeta" in wide else np.full(slots.size, np.nan)
    heat = wide["wvpts"].reindex(slots).to_numpy(float) if "wvpts" in wide else np.full(slots.size, np.nan)
    return _split(zeta, heat, scheme)


def _split(x: np.ndarray, heat: np.ndarray, scheme: TauScheme) -> tuple[np.ndarray, float]:
    """Neutral by |x|, the rest unstable where `heat` > 0, stable where < 0."""
    finite = np.isfinite(x) & np.isfinite(heat)
    neutral = np.zeros(x.size, dtype=bool)
    if scheme.local_mode == "zeta0":
        threshold = float(scheme.zeta0)
        neutral[finite] = np.abs(x[finite]) <= threshold
    else:  # exactly the share closest to 0, ties by slot order
        where = np.flatnonzero(finite)
        k = int(round(scheme.percent / 100.0 * where.size))
        order = where[np.argsort(np.abs(x[where]), kind="stable")]
        neutral[order[:k]] = True
        threshold = float(np.abs(x[order[k - 1]])) if k > 0 else 0.0
    codes = np.full(x.size, -1, dtype=np.int16)
    codes[finite & ~neutral & (heat > 0)] = 0
    codes[neutral] = 1
    codes[finite & ~neutral & (heat < 0)] = 2
    return codes, threshold


def boom_part(job: dict) -> dict:
    """One boom of a variant: its τ per slot, then secondary's statistics and
    derived quantities and tertiary's filtering on them, written to job["out"].
    """
    from ttu_tower.viewer.fragments import FragmentIndex
    boom, key, cfg = job["boom"], job["key"], job["cfg"]
    scheme = TauScheme.from_dict(job["scheme"])
    out = Path(job["out"])
    index = FragmentIndex(Path(job["run_dir"]))
    index.variants = StaticVariants(job["deps"])
    f = pa_dataset.field
    data = scheme.data

    stored = _selection_rows(index, boom, data)
    if stored.empty:
        return {"rows": 0}
    slots = stored["slot"].to_numpy(np.int64)
    base = stored if scheme.base == "run" else _whatif_rows(index, boom, scheme, cfg).set_index("slot") \
        .reindex(slots).reset_index()

    threshold = None
    if scheme.classifier == "global":
        codes = job["global_classes"].reindex(slots).fillna(-1).to_numpy(np.int16)
    elif scheme.classifier == "local":
        codes, threshold = _local_classes(index, boom, scheme, slots)
    else:
        codes = np.full(slots.size, -1, dtype=np.int16)
    names = scheme.class_names()
    picked = np.array([scheme.choices[c] if c >= 0 else scheme.otherwise for c in codes], dtype=object)
    rule = np.array([names[c] if c >= 0 else ("otherwise" if names else "all") for c in codes], dtype=object)

    sel = pd.DataFrame({"slot": slots, "boom": boom, "variant": data})
    sel["tau_s"], sel["source"], sel["source_status"] = np.nan, None, None
    for choice in np.unique(picked):
        rows = picked == choice
        if choice in ("base", "run"):
            src = base if choice == "base" else stored
            for c in ("tau_s", "source", "source_status"):
                sel.loc[rows, c] = src[c].to_numpy()[rows]
        else:
            sel.loc[rows, ["tau_s", "source", "source_status"]] = [float(choice), "fixed", "fixed"]
    sel["tau_s"] = sel["tau_s"].astype(float)

    # the ladder rows at the chosen rungs (unexcised: the mrd rows where primary didn't compute its own)
    rungs = sorted({float(t) for t in sel["tau_s"] if np.isfinite(t)})
    reads = ["mrd"] if data == "mrd" else ["mrd", "mrd_unexcised"]
    flt = (f("boom") == boom) & f("variant").isin(reads) & f("rung_s").isin(rungs)
    ladder = _obj(index.read("ladder", booms=[boom], filter=flt), ("variant", "variable", "stat"))
    ladder_cov = _obj(index.read("ladder_coverage", booms=[boom], filter=flt), ("variant", "family"))
    if data == "mrd_unexcised":
        sb = index.read("slot_boom", booms=[boom], columns=["slot", "status", "unexcised_computed"],
                        filter=f("boom") == boom)
        copy = set(sb.loc[(sb["status"].astype(object) == "computed") & ~sb["unexcised_computed"].astype(bool),
                          "slot"])
        frames = []
        for df in (ladder, ladder_cov):
            own = df[df["variant"] == "mrd_unexcised"]
            mine = df[(df["variant"] == "mrd") & df["slot"].isin(copy)].assign(variant="mrd_unexcised")
            frames.append(pd.concat([own, mine], ignore_index=True))
        ladder, ladder_cov = frames
    slow = _obj(index.read("slow", filter=f("boom") == boom), ("variable",))
    p_factor = slow[slow["variable"] == "p_factor"].set_index(["slot", "boom"])["value"]

    stats = selected_stats(ladder, ladder_cov, sel, p_factor)
    extra, labels = derived(stats, sel, slow, cfg)
    boom_stats = pd.concat([stats, extra], ignore_index=True).assign(variant=key)
    labels = labels.assign(variant=key)
    sel = sel.assign(variant=key)

    flags = index.read("flags", booms=[boom], columns=["test", "boom", "variable", "start", "end"],
                       filter=(f("boom") == boom) & f("test").isin(["bounds", "spike"]))
    no_coverage = pd.DataFrame({"slot": pd.Series(dtype=np.int64), "boom": pd.Series(dtype=np.int64),
                                "variable": pd.Series(dtype=object), "layer": pd.Series(dtype=object),
                                "fraction": pd.Series(dtype=float)})
    fail_keys, filter_log = filtering.evaluate(boom_stats, no_coverage, sel, {boom: FlagStore(flags)}, cfg.tertiary)
    boom_final = filtering.apply(boom_stats, fail_keys)
    labels_final = filtering.apply_to_labels(labels, fail_keys)

    starts = _slot_starts(job["run_dir"])
    # boom_stats only where filtering removed a value (the rest is boom_final's): half the disk
    _write(out, "boom_stats", boom, boom_stats[boom_final["value"].isna().to_numpy() & boom_stats["value"].notna().to_numpy()])
    _write(out, "boom_labels", boom, labels)
    _write(out, "tau_selected", boom, sel)
    _write(out, "boom_final", boom, boom_final, starts)
    _write(out, "boom_labels_final", boom, labels_final, starts)
    _write(out, "tau_final", boom, sel, starts)
    _write(out, "filter_log", boom, filter_log)
    _write(out, "tau_rule", boom, pd.DataFrame({"slot": slots, "boom": boom, "rule": rule}))

    counts = pd.crosstab(pd.Series(rule, name="rule"), pd.Series(sel["tau_s"].fillna(-1.0).to_numpy(), name="tau"))
    return {"rows": int(slots.size), "threshold": threshold,
            "counts": {r: {f"{t:g}": int(n) for t, n in row.items() if n} for r, row in counts.iterrows()}}


def _gradients(run, index, key: str, out: Path, cfg) -> None:
    """The variant's flux products of tertiary.gradients, added to each boom's boom_final: they
    need every boom, so they come once all are written.
    """
    f = pa_dataset.field
    columns = ["slot", "boom", "variant", "variable", "stat", "value"]
    flux = io_store.read_paths(paths(out, "boom_final"), columns=columns,
                               filter=f("variable").isin(list(gradients.FLUX_INPUTS)))
    none = index.read("boom_final", columns=columns,
                      filter=(f("variant") == "none") & f("variable").isin(list(gradients.NONE_INPUTS)))
    both = _obj(pd.concat([flux, none], ignore_index=True), ("variant", "variable", "stat"))
    rows = gradients.gradients(both, cfg.files.booms, cfg.tertiary.gradients.stencils)  # tertiary's stencils exactly
    rows = rows[rows["variant"] == key]
    starts = _slot_starts(run.run_dir)
    for boom, part in rows.groupby("boom"):
        final = io_store.read_paths(paths(out, "boom_final", [boom]))
        if final.empty:
            continue
        final = _obj(final.drop(columns="slot_start"), ("variant", "variable", "stat"))
        _write(out, "boom_final", int(boom), pd.concat([final, part], ignore_index=True), starts)


def _profile(run, index, key: str, out: Path, cfg) -> None:
    f = pa_dataset.field
    final = io_store.read_paths(paths(out, "boom_final"), columns=["slot", "boom", "variant", "variable", "stat", "value"],
                                filter=f("variable").isin(["ti", "wd", "ustar"]))
    ws = index.read("boom_final", columns=["slot", "boom", "variant", "variable", "stat", "value"],
                    filter=(f("variant") == "none") & (f("variable") == "ws") & (f("stat") == "mean"))
    both = _obj(pd.concat([final, ws], ignore_index=True), ("variant", "variable", "stat"))
    tau = _obj(io_store.read_paths(paths(out, "tau_selected")), ("variant", "source", "source_status"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slots, as in tertiary's own run
        rows = profiles.variant_profiles(both, tau, key, cfg.tertiary.fits)
    _write(out, "profile", -1, rows, _slot_starts(run.run_dir))
