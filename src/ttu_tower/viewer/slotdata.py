"""Everything the run stored about one slot, read from the few fragment
files that hold it, plus what the inspector derives from it without raw data:
each (co)spectrum's arrays, the tau detection rerun with its trace, and the
integral scales at every rung.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyarrow.dataset as pa_dataset

from ttu_tower import schema
from ttu_tower.secondary.detect import SMOOTHING, Detection, detect
from ttu_tower.secondary.select import select
from ttu_tower.viewer import tauvariants
from ttu_tower.viewer.fragments import FragmentIndex

COSPECTRA = ("wvpts", "uw", "vw", "uv")
VARIANCES = ("uu", "vv", "ww", "tsts")
SPECTRA = COSPECTRA + VARIANCES
DETECTED = {"heat": "wvpts", "momentum": "uw"}
SPECTRUM_LABELS = {"wvpts": "w'θv'", "uw": "u'w'", "vw": "v'w'", "uv": "u'v'", "uu": "u'u'", "vv": "v'v'",
                   "ww": "w'w'", "tsts": "Ts'Ts'"}
SPECTRUM_UNITS = {"wvpts": "K m/s", "uw": "m²/s²", "vw": "m²/s²", "uv": "m²/s²", "uu": "m²/s²", "vv": "m²/s²",
                  "ww": "m²/s²", "tsts": "K²"}
NEIGHBOURS = 3
_SLOT_TABLES = ("slot_boom", "coverage", "means", "slot_qc", "mrd_frame", "ladder", "ladder_coverage", "tau",
                "tau_selected", "boom_stats", "boom_labels", "slow", "slot_stats", "boom_final", "tau_final",
                "pairs", "profile", "slot_final", "filter_log", "boom_labels_final")


@dataclass
class SlotBundle:
    slot: int
    boom: int
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)  # every stored row of this slot
    mrd: pd.DataFrame | None = None  # this slot, every boom; plus the boom's neighbouring slots
    te: pd.DataFrame | None = None  # transport efficiency per rung, over the same slots and booms as mrd
    ladder: pd.DataFrame | None = None
    ladder_coverage: pd.DataFrame | None = None

    def table(self, name: str) -> pd.DataFrame:
        df = self.tables.get(name)
        return df if df is not None else pd.DataFrame(columns=list(schema.TABLES[name].columns))


def _stage_present(index: FragmentIndex, table: str) -> bool:
    return index.table(table).table_dir.is_dir()


def load_slot(index: FragmentIndex, k: int, boom: int, booms=range(1, 11)) -> SlotBundle:
    """Every table's rows for slot k (the boom's rows, where a table has booms),
    the slot's spectra for every boom, and the boom's spectra for the slots
    around it.
    """
    f = pa_dataset.field
    bundle = SlotBundle(slot=k, boom=boom)
    h = k // 3
    for name in _SLOT_TABLES:
        if not _stage_present(index, name):
            continue
        cols = schema.TABLES[name].columns
        flt = f("slot") == k
        if "boom" in cols and name not in ("pairs",):
            flt = flt & (f("boom") == boom)
        per_boom = schema.TABLES[name].stage == "primary"
        bundle.tables[name] = index.read(name, booms=[boom] if per_boom else None, half_hours=[h], filter=flt)

    if _stage_present(index, "mrd"):
        this_slot = index.read("mrd", booms=list(booms), half_hours=[h], filter=f("slot") == k)
        around = index.read("mrd", booms=[boom], half_hours=range((k - NEIGHBOURS) // 3, (k + NEIGHBOURS) // 3 + 1),
                            filter=(f("slot") >= k - NEIGHBOURS) & (f("slot") <= k + NEIGHBOURS) & (f("slot") != k))
        bundle.mrd = pd.concat([this_slot, around], ignore_index=True)
        for col in ("variant", "spectrum"):
            bundle.mrd[col] = bundle.mrd[col].astype(str)
    if _stage_present(index, "ladder"):
        te = index.read("ladder", booms=list(booms), half_hours=range((k - NEIGHBOURS) // 3, (k + NEIGHBOURS) // 3 + 1),
                        columns=["slot", "boom", "variant", "rung_s", "variable", "value"],
                        filter=(f("stat") == "te") & (f("slot") >= k - NEIGHBOURS) & (f("slot") <= k + NEIGHBOURS))
        te = te[(te["slot"] == k) | (te["boom"] == boom)].reset_index(drop=True)
        for col in ("variant", "variable"):
            te[col] = te[col].astype(str)
        bundle.te = te
    bundle.ladder = bundle.tables.get("ladder")
    bundle.ladder_coverage = bundle.tables.get("ladder_coverage")
    for df in bundle.tables.values():
        _plain(df)
    return bundle


_KEY_COLUMNS = ("variant", "variable", "stat", "spectrum", "cospectrum", "family", "layer", "status", "source",
                "source_status", "label", "group", "criterion", "test", "reversal_type")


def _plain(df: pd.DataFrame) -> pd.DataFrame:
    """Categorical key columns as plain objects, so rows compare and filter as strings."""
    for col in _KEY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(object)
    return df


_ROW_KEYS = ("boom", "boom2", "variant", "rung_s", "spectrum", "cospectrum", "scale_s", "family", "variable", "stat",
             "layer", "label", "group", "criterion")


def _key_text(v) -> str:
    return f"{v:g}" if isinstance(v, (float, np.floating)) else str(v)


def all_rows(tables: dict[str, pd.DataFrame], boom: int) -> pd.DataFrame:
    """Every stored value of the slot, one per line: its table, the key that
    picks its row, its column (`field`), the value, and the row's position in
    its table. A row with no value columns (filter_log) is one line with an
    empty field.
    """
    parts = []
    for name, df in tables.items():
        drop = {"slot", "slot_start"}
        if "boom" in df.columns and (df["boom"] == boom).all():
            drop.add("boom")
        keys = [c for c in df.columns if c in _ROW_KEYS and c not in drop]
        fields = [c for c in df.columns if c not in drop and c not in keys]
        key = pd.Series("", index=df.index, dtype=object)
        for i, col in enumerate(keys):
            key = key + ("" if i == 0 else ", ") + col + "=" + df[col].astype(object).map(_key_text)
        row = np.arange(len(df))
        lines = [pd.DataFrame({"table": name, "key": key.to_numpy(), "field": f, "value": df[f].astype(object).to_numpy(),
                               "row": row}) for f in fields]
        if not fields:
            lines = [pd.DataFrame({"table": name, "key": key.to_numpy(), "field": "", "value": "", "row": row})]
        parts.append(pd.concat(lines, ignore_index=True).sort_values("row", kind="stable"))
    if not parts:
        return pd.DataFrame(columns=["table", "key", "field", "value", "row"])
    return pd.concat(parts, ignore_index=True)


# --- the slot across every boom -------------------------------------------------------------

_ACROSS_TABLES = ("slot_boom", "coverage", "means", "ladder", "tau", "tau_selected", "boom_stats", "slow",
                  "boom_labels", "boom_final", "tau_final", "pairs", "profile", "filter_log")


@dataclass
class SlotAcross:
    """One slot's rows for every boom: what the profile and the tau what-if need."""
    slot: int
    booms: tuple[int, ...]
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    mrd: pd.DataFrame | None = None

    def table(self, name: str) -> pd.DataFrame:
        df = self.tables.get(name)
        return df if df is not None else pd.DataFrame(columns=list(schema.TABLES[name].columns))


def load_across(index: FragmentIndex, k: int, booms) -> SlotAcross:
    f = pa_dataset.field
    booms = tuple(booms)
    across = SlotAcross(slot=k, booms=booms)
    h = k // 3
    for name in _ACROSS_TABLES:
        if not _stage_present(index, name):
            continue
        per_boom = schema.TABLES[name].stage == "primary"
        across.tables[name] = _plain(index.read(name, booms=list(booms) if per_boom else None, half_hours=[h],
                                                filter=f("slot") == k))
    if _stage_present(index, "mrd"):
        across.mrd = _plain(index.read("mrd", booms=list(booms), half_hours=[h], filter=f("slot") == k))
    return across


# --- spectra --------------------------------------------------------------------------------

@dataclass
class Spectrum:
    scale_s: np.ndarray
    value: np.ndarray
    se: np.ndarray
    n_pairs: np.ndarray

    @property
    def ogive(self) -> np.ndarray:
        """Cumulative sum from the smallest scale: the mean within-block
        (co)variance at each scale, where every block is valid.
        """
        return np.nancumsum(self.value)


NORMALIZE_TIP = ("divide by the sum over every mode (the within-block (co)variance at the largest scale), "
                 "as the Summary's composite spectra do; its ± SE and detection curves alike")


def normalizing_factor(spec: Spectrum | None) -> float:
    """1 / the spectrum's sum over modes; NaN where that's 0 or there are no values."""
    if spec is None:
        return np.nan
    total = np.nansum(spec.value)
    return 1.0 / total if np.isfinite(total) and total != 0 else np.nan


def normalized(spec: Spectrum | None) -> Spectrum | None:
    if spec is None:
        return None
    f = normalizing_factor(spec)
    return Spectrum(spec.scale_s, spec.value * f, spec.se * abs(f), spec.n_pairs)


def spectrum(mrd: pd.DataFrame | None, slot: int, boom: int, variant: str, name: str) -> Spectrum | None:
    if mrd is None or mrd.empty:
        return None
    rows = mrd[(mrd["slot"] == slot) & (mrd["boom"] == boom) & (mrd["variant"] == variant) & (mrd["spectrum"] == name)]
    if rows.empty:
        return None
    rows = rows.sort_values("scale_s")
    return Spectrum(rows["scale_s"].to_numpy(float), rows["value"].to_numpy(float), rows["se"].to_numpy(float),
                    rows["n_pairs"].to_numpy())


def transport_efficiency(te: pd.DataFrame | None, slot: int, boom: int, variant: str,
                         variable: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(rung_s, TE) of `variable` ("uw" or "wvpts"), or None without rows."""
    if te is None or te.empty:
        return None
    rows = te[(te["slot"] == slot) & (te["boom"] == boom) & (te["variant"] == variant) & (te["variable"] == variable)]
    if rows.empty:
        return None
    rows = rows.sort_values("rung_s")
    return rows["rung_s"].to_numpy(float), rows["value"].to_numpy(float)


def spectrum_variant(bundle: SlotBundle, variant: str) -> str:
    """The mrd variant holding `variant`'s spectra: naive shares mrd's, and an
    mrd_unexcised primary never computed (nothing was excised) equals mrd.
    """
    variant = tauvariants.basis(variant)  # naive's and the viewer's variants': the data they're computed from
    if variant == "mrd_unexcised" and bundle.mrd is not None:
        has = ((bundle.mrd["slot"] == bundle.slot) & (bundle.mrd["boom"] == bundle.boom)
               & (bundle.mrd["variant"] == "mrd_unexcised")).any()
        return "mrd_unexcised" if has else "mrd"
    return variant


def family_has_data(bundle: SlotBundle, variant: str, family: str) -> bool:
    """The same test secondary applies before detecting: any usable (or, for
    mrd_unexcised, unexcised) sample of the family in the slot.
    """
    cov = bundle.table("coverage")
    layer = "unexcised" if variant == "mrd_unexcised" else "usable"
    row = cov[(cov["variable"] == family) & (cov["layer"] == layer)]
    return bool(not row.empty and row["fraction"].iloc[0] > 0)


@dataclass
class DetectionView:
    family: str
    spectrum: str
    detection: Detection  # rerun now, on the stored spectrum, with the run's detection config
    trace: dict
    stored: dict | None  # the run's own tau row for this cospectrum, if any

    @property
    def agrees(self) -> bool:
        if self.stored is None:
            return True
        same_status = self.stored["status"] == self.detection.status
        a, b = self.stored["tau_s"], self.detection.tau_s
        return same_status and ((np.isnan(a) and np.isnan(b)) or a == b)


def rerun_detection(bundle: SlotBundle, variant: str, cfg_secondary,
                    kernel: tuple[float, ...] = SMOOTHING) -> tuple[dict[str, DetectionView], tuple]:
    """Each detected cospectrum's detection (with trace) and the selection,
    rerun on the stored spectra: (views by family, (tau_s, source, source_status)).
    `kernel`: detection's smoothing weights.
    """
    mrd_variant = spectrum_variant(bundle, variant)
    tau = bundle.table("tau")
    views = {}
    for family, name in DETECTED.items():
        spec = spectrum(bundle.mrd, bundle.slot, bundle.boom, mrd_variant, name)
        trace: dict = {}
        if spec is None:
            det = detect(np.zeros(0), np.zeros(0), np.zeros(0), False, cfg_secondary.detection, trace=trace,
                         kernel=kernel)
        else:
            det = detect(spec.value, spec.se, spec.n_pairs, family_has_data(bundle, mrd_variant, family),
                         cfg_secondary.detection, trace=trace, kernel=kernel)
        stored = tau[(tau["variant"] == mrd_variant) & (tau["cospectrum"] == family)]
        views[family] = DetectionView(family, name, det, trace, stored.iloc[0].to_dict() if not stored.empty else None)
    selection = select(views["heat"].detection, views["momentum"].detection, cfg_secondary.selection)
    return views, selection


def stored_selection(bundle: SlotBundle, variant: str) -> dict | None:
    sel = bundle.table("tau_selected")
    row = sel[sel["variant"] == variant]
    return row.iloc[0].to_dict() if not row.empty else None


# --- scales -----------------------------------------------------------------------------------

def rung_scales(bundle: SlotBundle, variant: str) -> pd.DataFrame:
    """Per rung: ITS of u, v, w and vpts; ILS = ITS × |u mean| at that rung
    (as secondary defines it); and the fully usable blocks behind each ITS.
    """
    ladder = bundle.ladder
    variant = tauvariants.basis(variant)
    if ladder is None or ladder.empty:
        return pd.DataFrame()
    lad = ladder[ladder["variant"] == variant]
    if lad.empty and variant == "mrd_unexcised":
        lad = ladder[ladder["variant"] == "mrd"]
    its = lad[lad["stat"] == "its"].pivot_table(index="rung_s", columns="variable", values="value")
    its.columns = [f"its_{c}" for c in its.columns]
    u_mean = lad[(lad["variable"] == "u") & (lad["stat"] == "mean")].set_index("rung_s")["value"].abs()
    out = its.copy()
    for var in ("u", "v", "w", "vpts"):
        if f"its_{var}" in out.columns:
            out[f"ils_{var}"] = out[f"its_{var}"] * u_mean.reindex(out.index)
    ordered = [f"{p}_{v}" for p in ("its", "ils") for v in ("u", "v", "w", "vpts")]
    out = out[[c for c in ordered if c in out.columns]]
    cov = bundle.ladder_coverage
    if cov is not None and not cov.empty:
        cov = cov[(cov["variant"] == (variant if (cov["variant"] == variant).any() else "mrd"))]
        for fam in ("its", "its_vpts"):
            rows = cov[cov["family"] == fam].set_index("rung_s")
            out[f"{fam}_blocks"] = (rows["blocks_used"].astype(str) + "/" + rows["blocks_total"].astype(str)).reindex(
                out.index)
    return out.sort_index()
