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
from ttu_tower.secondary.detect import Detection, detect
from ttu_tower.secondary.select import select
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


def spectrum(mrd: pd.DataFrame | None, slot: int, boom: int, variant: str, name: str) -> Spectrum | None:
    if mrd is None or mrd.empty:
        return None
    rows = mrd[(mrd["slot"] == slot) & (mrd["boom"] == boom) & (mrd["variant"] == variant) & (mrd["spectrum"] == name)]
    if rows.empty:
        return None
    rows = rows.sort_values("scale_s")
    return Spectrum(rows["scale_s"].to_numpy(float), rows["value"].to_numpy(float), rows["se"].to_numpy(float),
                    rows["n_pairs"].to_numpy())


def spectrum_variant(bundle: SlotBundle, variant: str) -> str:
    """The mrd variant holding `variant`'s spectra: naive shares mrd's, and an
    mrd_unexcised primary never computed (nothing was excised) equals mrd.
    """
    if variant == "naive":
        return "mrd"
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


def rerun_detection(bundle: SlotBundle, variant: str, cfg_secondary) -> tuple[dict[str, DetectionView], tuple]:
    """Each detected cospectrum's detection (with trace) and the selection,
    rerun on the stored spectra: (views by family, (tau_s, source, source_status)).
    """
    mrd_variant = spectrum_variant(bundle, variant)
    tau = bundle.table("tau")
    views = {}
    for family, name in DETECTED.items():
        spec = spectrum(bundle.mrd, bundle.slot, bundle.boom, mrd_variant, name)
        trace: dict = {}
        if spec is None:
            det = detect(np.zeros(0), np.zeros(0), np.zeros(0), False, cfg_secondary.detection, trace=trace)
        else:
            det = detect(spec.value, spec.se, spec.n_pairs, family_has_data(bundle, mrd_variant, family),
                         cfg_secondary.detection, trace=trace)
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
    variant = "mrd" if variant == "naive" else variant
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
