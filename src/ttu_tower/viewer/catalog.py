"""Every plottable quantity of a run, built from the run's own table
vocabularies (so nothing listed can be missing, and nothing present is
hidden), with labels, units and groups for display.
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ttu_tower.flags import TESTS
from ttu_tower.io.runs import locate_home

VARIANTS = ("mrd", "naive", "mrd_unexcised")
_CACHE_VERSION = 1

# variable -> (group, label, unit); stat-specific labels are composed below
_META: dict[str, tuple[str, str, str]] = {
    "ue": ("Wind", "ue (east)", "m/s"), "vn": ("Wind", "vn (north)", "m/s"), "w": ("Wind", "w", "m/s"),
    "ws": ("Wind", "wind speed", "m/s"), "wd": ("Wind", "wind direction", "deg"), "u": ("Turbulence", "u", "m/s"),
    "v": ("Turbulence", "v", "m/s"),
    "uv": ("Fluxes", "u'v'", "m²/s²"), "uw": ("Fluxes", "u'w'", "m²/s²"), "vw": ("Fluxes", "v'w'", "m²/s²"),
    "wvpts": ("Fluxes", "w'θv'", "K m/s"),
    "ts": ("Thermo", "sonic T", "K"), "vpts": ("Thermo", "sonic θv", "K"),
    "t": ("Thermo", "T", "K"), "rh": ("Thermo", "RH", "fraction"), "p": ("Thermo", "p", "kPa"),
    "es": ("Thermo", "e_s", "kPa"), "e": ("Thermo", "e", "kPa"), "r": ("Thermo", "mixing ratio", "kg/kg"),
    "q": ("Thermo", "specific humidity", "kg/kg"), "td": ("Thermo", "dewpoint", "K"), "pt": ("Thermo", "θ", "K"),
    "vt": ("Thermo", "Tv", "K"), "vpt": ("Thermo", "θv", "K"), "p_factor": ("Thermo", "pressure factor", ""),
    "p_measured": ("QC", "p measured", "0/1"),
    "sigma_u": ("Turbulence", "σu", "m/s"), "sigma_v": ("Turbulence", "σv", "m/s"), "sigma_w": ("Turbulence", "σw", "m/s"),
    "ti": ("Turbulence", "TI", ""), "ti_u": ("Turbulence", "TI u", ""), "ti_v": ("Turbulence", "TI v", ""),
    "ti_w": ("Turbulence", "TI w", ""), "ti_ratio_vu": ("Turbulence", "TI v/u", ""),
    "ti_ratio_wu": ("Turbulence", "TI w/u", ""), "tke": ("Turbulence", "TKE", "m²/s²"),
    "ctke": ("Turbulence", "cTKE", "m²/s²"), "ustar": ("Fluxes", "u*", "m/s"),
    "obukhov_length": ("Stability", "Obukhov L", "m"), "zeta": ("Stability", "ζ = z/L", ""),
    "ils_u": ("Scales", "ILS u", "m"), "ils_v": ("Scales", "ILS v", "m"), "ils_w": ("Scales", "ILS w", "m"),
    "ils_vpts": ("Scales", "ILS θv", "m"), "ils_ratio_vu": ("Scales", "ILS v/u", ""),
    "ils_ratio_wu": ("Scales", "ILS w/u", ""), "its_ratio_u": ("Scales", "τ/ITS u", ""),
    "its_ratio_v": ("Scales", "τ/ITS v", ""), "its_ratio_w": ("Scales", "τ/ITS w", ""),
    "its_ratio_vpts": ("Scales", "τ/ITS θv", ""), "its_short": ("Scales", "ITS short (u,v,w)", "0/1"),
    "its_short_vpts": ("Scales", "ITS short (θv)", "0/1"),
    "aniso_l1": ("Turbulence", "anisotropy λ1", ""), "aniso_l2": ("Turbulence", "anisotropy λ2", ""),
    "aniso_l3": ("Turbulence", "anisotropy λ3", ""), "aniso_beta": ("Turbulence", "anisotropy β", "deg"),
    "aniso_phi": ("Turbulence", "anisotropy φ", "deg"), "w_ratio": ("QC", "w/ws (tilt)", ""),
    "momentum": ("QC", "momentum family", ""), "heat": ("QC", "heat family", ""), "its_vpts": ("QC", "ITS θv blocks", ""),
    "rib": ("Stability", "Ri_b", ""), "lapse_vpt": ("Stability", "dθv/dz", "K/m"), "veer": ("Stability", "veer", "deg"),
    "alpha": ("Profile fits", "α (ws power law)", ""), "loglaw": ("Profile fits", "log-law fit", ""), "gamma": ("Profile fits", "γ (TI power law)", ""),
    "wdgamma": ("Profile fits", "γ (wd std power law)", ""), "loglaw_ustar": ("Profile fits", "log-law u*", "m/s"),
    "loglaw_z0": ("Profile fits", "log-law z0", "m"), "loglaw_z0_constrained": ("Profile fits", "log-law z0 (u* fixed)", "m"),
    "sun_elevation": ("Other", "sun elevation", "deg"),
    "solar": ("Mesonet", "solar radiation", "W/m²"), "precip": ("Mesonet", "precipitation", "mm"),
    "ti_meso": ("Mesonet", "mesonet TI", ""),
}
_STAT_LABELS = {"mean": "mean", "var": "variance", "cov": "covariance", "std": "std", "its": "ITS", "te": "transport eff.",
                "max": "max", "vector_mean": "vector mean", "unit_mean": "unit-vector mean", "coverage": "coverage",
                "blocks_used": "blocks used", "blocks_total": "blocks total", "skew": "skewness", "kurt": "kurtosis",
                "mean_l1": "first-layer mean", "std_l1": "first-layer std"}
_CIRCULAR = {"wd", "veer", "wd_deg", "wd_meso_mean"}
_GROUP_ORDER = ("Wind", "Turbulence", "Fluxes", "Scales", "Stability", "Thermo", "Profile fits", "τ", "QC", "Mesonet", "Other")


@dataclass(frozen=True)
class Quantity:
    key: str
    table: str
    variable: str
    stat: str | None
    kind: str  # "boom" | "pair" | "slot"
    variants: tuple[str, ...]
    label: str
    unit: str
    group: str
    column: str = "value"
    extra: tuple[tuple[str, str], ...] = ()  # further equality filters, e.g. (("layer", "usable"),)
    pairs: tuple[tuple[int, int], ...] = ()
    circular: bool = False
    categorical: bool = False

    @property
    def title(self) -> str:
        return f"{self.label} [{self.unit}]" if self.unit else self.label


def _describe(variable: str, stat: str | None) -> tuple[str, str, str]:
    """(group, label, unit) for a (variable, stat) pair."""
    if "_meso" in variable:
        _, _, unit = _META.get(variable.split("_meso")[0], ("", "", ""))
        return "Mesonet", f"mesonet {variable.replace('_meso', '').replace('_', ' ')}", unit
    group, label, unit = _META.get(variable, ("Other", variable, ""))
    if stat is None:
        return group, label, unit

    stat_label = _STAT_LABELS.get(stat)
    if stat_label is None:  # gust stats: max_{p}s / std_{p}s
        kind, _, period = stat.partition("_")
        stat_label = f"{kind} of {period} means" if period else stat
    if stat == "its":
        return "Scales", f"ITS {label}", "s"
    if stat in ("coverage", "blocks_used", "blocks_total"):
        return "QC", f"{label} {stat_label}", ""
    if stat == "var" and unit:
        unit = f"({unit})²"
    elif stat in ("te", "skew", "kurt"):
        unit = ""
    if stat == "cov":  # "u'w'" already names a covariance
        return group, label, unit
    if group == "Wind" and stat in ("var", "std"):
        group = "Turbulence"
    return group, f"{label} {stat_label}", unit


def variable_label(variable: str) -> tuple[str, str]:
    """(label, unit) of a bare variable, e.g. a 50-Hz series."""
    _, label, unit = _META.get(variable, ("", variable, ""))
    return label, unit


def _distinct(paths: list[Path], columns: list[str]) -> set[tuple]:
    """Distinct value combinations of `columns` across fragment files, from
    their dictionary codes (fast; no per-row Python objects).
    """
    out: set[tuple] = set()
    for path in paths:
        table = pq.read_table(path, columns=columns)
        if table.num_rows == 0:
            continue
        codes, values = [], []
        for name in columns:
            arr = table[name].combine_chunks()
            if pa.types.is_dictionary(arr.type):
                idx = arr.indices.fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int64)
                codes.append(idx)
                values.append(arr.dictionary.to_pylist())
            else:
                raw = arr.to_numpy(zero_copy_only=False)
                uniq, inv = np.unique(raw, return_inverse=True)
                codes.append(inv.astype(np.int64))
                values.append([v.item() if hasattr(v, "item") else v for v in uniq])
        # one int64 key per row (each code shifted by 1 so a null's -1 becomes 0)
        radices = [len(v) + 1 for v in values]
        key = np.zeros(table.num_rows, dtype=np.int64)
        for c, radix in zip(codes, radices):
            key = key * radix + (c + 1)
        for k in np.unique(key):
            row = []
            for radix in reversed(radices):
                k, digit = divmod(int(k), radix)
                row.append(digit - 1)
            row.reverse()
            out.add(tuple(values[i][c] if c >= 0 else None for i, c in enumerate(row)))
    return out


_VOCAB_TABLES = {
    "boom_final": ["variant", "variable", "stat"],
    "pairs": ["variant", "variable", "boom", "boom2"],
    "profile": ["variant", "variable"],
    "slot_final": ["variable"],
    "boom_labels_final": ["variant", "label"],
}


def scan_vocabulary(run) -> dict[str, list[list]]:
    """Distinct key combinations of each tertiary table, cached under the
    ttu-tower home (never inside the run directory).
    """
    tertiary = run.stage_dir("tertiary")
    summary = tertiary / "run_summary.json"
    stamp = f"{run.meta['tertiary'].get('config_hash') if run.meta['tertiary'] else 'none'}-" \
            f"{int(summary.stat().st_mtime) if summary.is_file() else 0}"
    home = locate_home()
    cache_path = home / "viewer" / "cache" / f"{run.tag}-{stamp}-v{_CACHE_VERSION}.json" if home is not None else None
    if cache_path is not None and cache_path.is_file():
        with open(cache_path) as f:
            return json.load(f)

    vocab = {}
    for table, columns in _VOCAB_TABLES.items():
        paths = sorted((tertiary / "data" / table).glob("*.parquet"))
        vocab[table] = sorted((list(r) for r in _distinct(paths, columns)), key=lambda r: [str(v) for v in r])
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(vocab, f)
    return vocab


@dataclass
class Catalog:
    quantities: dict[str, Quantity] = field(default_factory=dict)

    def add(self, q: Quantity) -> None:
        self.quantities[q.key] = q

    def __getitem__(self, key: str) -> Quantity:
        return self.quantities[key]

    def __iter__(self):
        return iter(self.quantities.values())

    def __len__(self) -> int:
        return len(self.quantities)

    def grouped(self) -> dict[str, list[Quantity]]:
        groups: dict[str, list[Quantity]] = {}
        for q in self.quantities.values():
            groups.setdefault(q.group, []).append(q)
        order = {g: i for i, g in enumerate(_GROUP_ORDER)}
        return {g: sorted(groups[g], key=lambda q: q.label.lower())
                for g in sorted(groups, key=lambda g: order.get(g, len(order)))}


def build_catalog(run) -> Catalog:
    cat = Catalog()
    if "tertiary" in run.stages:
        _add_tertiary(cat, scan_vocabulary(run))
    _add_long_tables(cat, run)
    return cat


def _add_tertiary(cat: Catalog, vocab: dict) -> None:
    by_key: dict[tuple, set[str]] = {}
    for variant, variable, stat in vocab["boom_final"]:
        by_key.setdefault((variable, stat), set()).add(variant)
    for (variable, stat), variants in by_key.items():
        group, label, unit = _describe(variable, stat)
        if stat is None and variable in ("t", "rh", "p"):
            label = f"{label} (slow table)"  # the same value as the stat="mean" row, via secondary's slow table
        elif stat is None and variable == "vpts":
            label = f"{label} mean, pressure-corrected"  # the stat="mean" row is at the reference pressure
        ordered = tuple(v for v in ("none",) + VARIANTS if v in variants)
        cat.add(Quantity(key=f"boom_final|{variable}|{stat or ''}", table="boom_final", variable=variable, stat=stat,
                         kind="boom", variants=ordered, label=label, unit=unit, group=group,
                         circular=variable in _CIRCULAR and stat in ("mean", "unit_mean")))

    pairs: dict[tuple, set] = {}
    for variant, variable, boom, boom2 in vocab["pairs"]:
        pairs.setdefault((variable, variant), set()).add((int(boom), int(boom2)))
    for (variable, variant), combos in pairs.items():
        group, label, unit = _describe(variable, None)
        cat.add(Quantity(key=f"pairs|{variable}", table="pairs", variable=variable, stat=None, kind="pair",
                         variants=(variant,), label=label, unit=unit, group=group, pairs=tuple(sorted(combos)),
                         circular=variable == "veer"))

    profile: dict[str, set[str]] = {}
    for variant, variable in vocab["profile"]:
        profile.setdefault(variable, set()).add(variant)
    for variable, variants in profile.items():
        base = variable.split("_n")[0] if variable.endswith(("_n", "_n_capped")) else variable
        group, label, unit = _describe(base, None)
        if base != variable:
            label, unit = f"{label} — booms {'capped' if variable.endswith('capped') else 'used'}", "count"
        ordered = tuple(v for v in ("none",) + VARIANTS if v in variants)
        cat.add(Quantity(key=f"profile|{variable}", table="profile", variable=variable, stat=None, kind="slot",
                         variants=ordered, label=label, unit=unit, group="Profile fits"))

    for (variable,) in vocab["slot_final"]:
        if variable == "night":  # drawn as shading instead
            continue
        group, label, unit = _describe(variable, None)
        cat.add(Quantity(key=f"slot_final|{variable}", table="slot_final", variable=variable, stat=None, kind="slot",
                         variants=("none",), label=label, unit=unit, group=group, circular=variable in _CIRCULAR))

    label_variants = tuple(v for v in VARIANTS if [v, "aniso_class"] in vocab["boom_labels_final"])
    if label_variants:
        cat.add(Quantity(key="boom_labels_final|aniso_class", table="boom_labels_final", variable="aniso_class",
                         stat=None, kind="boom", variants=label_variants, label="anisotropy class", unit="",
                         group="Turbulence", column="value", categorical=True))

    cat.add(Quantity(key="tau_final|tau_s", table="tau_final", variable="tau", stat=None, kind="boom",
                     variants=VARIANTS, label="selected τ", unit="s", group="τ", column="tau_s"))
    cat.add(Quantity(key="tau_final|source", table="tau_final", variable="tau source", stat=None, kind="boom",
                     variants=VARIANTS, label="τ source", unit="", group="τ", column="source", categorical=True))
    cat.add(Quantity(key="tau_final|source_status", table="tau_final", variable="tau status", stat=None,
                     kind="boom", variants=VARIANTS, label="τ status", unit="", group="τ", column="source_status",
                     categorical=True))


_COVERAGE_VARS = ("ue", "vn", "w", "ts", "vpts", "t", "rh", "p", "momentum", "heat")
_COVERAGE_LAYERS = ("present", "filled", "usable_l1", "usable", "unexcised")


def _add_long_tables(cat: Catalog, run) -> None:
    for cospectrum in ("heat", "momentum"):
        for column, label, unit, categorical in (("tau_s", "τ", "s", False), ("status", "status", "", True),
                                                 ("peak_scale_s", "peak scale", "s", False),
                                                 ("reversal_scale_s", "reversal scale", "s", False)):
            cat.add(Quantity(key=f"tau|{cospectrum}|{column}", table="tau", variable=f"{cospectrum} τ", stat=None,
                             kind="boom", variants=("mrd", "mrd_unexcised"), label=f"{cospectrum} cospectrum {label}",
                             unit=unit, group="τ", column=column, extra=(("cospectrum", cospectrum),),
                             categorical=categorical))

    for variable in _COVERAGE_VARS:
        layers = ("usable", "unexcised") if variable in ("momentum", "heat") else (
            ("present", "usable") if variable in ("t", "rh", "p") else _COVERAGE_LAYERS)
        for layer in layers:
            cat.add(Quantity(key=f"coverage|{variable}|{layer}", table="coverage", variable=variable, stat=None,
                             kind="boom", variants=("none",), label=f"{variable} coverage ({layer})", unit="fraction",
                             group="QC", column="fraction", extra=(("layer", layer),)))

    for variable in ("ue", "vn", "w", "ts"):
        for stat in ("skew", "kurt"):
            cat.add(Quantity(key=f"slot_qc|{variable}|{stat}", table="slot_qc", variable=variable, stat=stat,
                             kind="boom", variants=("none",), label=f"{variable} {_STAT_LABELS[stat]}", unit="",
                             group="QC"))
    for variable, stat in (("wd", "mean_l1"), ("ws", "mean_l1"), ("ws", "std_l1")):
        cat.add(Quantity(key=f"slot_qc|{variable}|{stat}", table="slot_qc", variable=variable, stat=stat,
                         kind="boom", variants=("none",), label=f"{variable} {_STAT_LABELS[stat]}",
                         unit="deg" if variable == "wd" else "m/s", group="QC", circular=variable == "wd"))

    cat.add(Quantity(key="mrd_frame|wd_deg", table="mrd_frame", variable="wd_deg", stat=None, kind="boom",
                     variants=("mrd", "mrd_unexcised"), label="detection-window wind direction", unit="deg",
                     group="Wind", column="wd_deg", circular=True))
    cat.add(Quantity(key="slot_boom|status", table="slot_boom", variable="status", stat=None, kind="boom",
                     variants=("none",), label="slot status", unit="", group="QC", column="status", categorical=True))

    for test, spec in TESTS.items():
        variables = spec.variables or ("ue",)
        for variable in variables:
            cat.add(Quantity(key=f"flags|{test}|{variable}", table="flags", variable=variable, stat=None, kind="boom",
                             variants=("none",), label=f"{test} flag fraction ({variable})", unit="fraction",
                             group="QC", column="fraction", extra=(("test", test),)))


def quantity_to_dict(q: Quantity) -> dict:
    return asdict(q)
