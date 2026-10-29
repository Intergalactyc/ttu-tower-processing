"""Per-boom vertical gradients and the quantities built on them, read from
already-filtered `boom_final` rows, so a filtered stencil point gives NaN.

Variant `none` (no τ dependence): dpt_dz, dvpt_dz, dws_dz (of the vector-mean
speed), shear (|d(ue, vn)/dz|), n2 = g/θv dθv/dz and rig = n2/shear². Each flux
variant: phi_m, phi_h (θ), phi_h_v (θv), rif, km, kh, prandtl = km/kh and
l_buoyant = σw/N. θv in n2 and rif is the boom's own (slow-sensor `vpt`, as in ζ),
g the local gravity at the boom, so prandtl = rig/rif exactly.

rif and km use the full shear production: the earth-frame wind gradient is
rotated into the boom's streamwise frame (its slot-mean wind) to meet uw and vw.

Every stencil in `[tertiary.gradients].stencils` gives a full set: the first under
these names, the others suffixed `_<stencil>`. A boom's stencil is among the run's
booms: "log" its two neighbours, "log_linear" those and the boom itself; the lowest
and highest booms use themselves and their one neighbour under both
(math.stats.log_gradient_weights).
"""
import numpy as np
import pandas as pd

from ttu_tower.constants import HEIGHTS, SITE_ELEVATION, SITE_LATITUDE
from ttu_tower.math.polar import rotate_streamwise, streamwise_angle
from ttu_tower.math.stats import log_gradient_weights
from ttu_tower.physics.constants import KAPPA
from ttu_tower.physics.earth import local_gravity

STENCILS = ("log_linear", "log")
PROFILE_VARIABLES = ("dpt_dz", "dvpt_dz", "dws_dz", "shear", "n2", "rig")
FLUX_VARIABLES = ("phi_m", "phi_h", "phi_h_v", "rif", "km", "kh", "prandtl", "l_buoyant")
# filtering.QUANTITY_GROUPS: the family a filtered input would come from (never filtered themselves)
GROUPS = {"dpt_dz": "vpt", "dvpt_dz": "vpt", "dws_dz": "momentum", "shear": "momentum", "n2": "vpt", "rig": "vpt",
          "phi_m": "momentum", "km": "momentum", "l_buoyant": "momentum",
          "phi_h": "heat", "phi_h_v": "heat", "rif": "heat", "kh": "heat", "prandtl": "heat"}

_COLUMNS = ["slot", "boom", "variant", "variable", "stat", "value"]
_GRAVITY = {boom: local_gravity(SITE_LATITUDE, SITE_ELEVATION + height) for boom, height in HEIGHTS.items()}
# input variable -> its stat
NONE_INPUTS = {"ue": "mean", "vn": "mean", "pt": None, "vpt": None}
FLUX_INPUTS = {"ustar": None, "theta_star": None, "sigma_w": None, "uw": "cov", "vw": "cov", "wvpts": "cov"}


def name(variable: str, stencil: str, stencils) -> str:
    """The output name of `variable` under `stencil`: plain for the first of `stencils`."""
    return variable if stencil == stencils[0] else f"{variable}_{stencil}"


def split(variable: str) -> tuple[str, str | None]:
    """(product, stencil) of a suffixed product name, else (variable, None)."""
    for stencil in sorted(STENCILS, key=len, reverse=True):
        base = variable.removesuffix(f"_{stencil}")
        if base != variable and base in GROUPS:
            return base, stencil
    return variable, None


def all_names() -> list[str]:
    """Every name a product can have, under any stencil."""
    return [n for v in GROUPS for n in (v, *(f"{v}_{s}" for s in STENCILS))]


def stencil(booms, form: str) -> dict[int, tuple[list[int], np.ndarray]]:
    """boom -> (the booms its gradient uses, their weights)."""
    if form not in STENCILS:
        raise ValueError(f"no stencil {form!r}")
    booms = sorted(booms)
    out = {}
    if len(booms) < 2:
        return out
    for i, boom in enumerate(booms):
        if i == 0 or i == len(booms) - 1:
            points = booms[:2] if i == 0 else booms[-2:]
        elif form == "log":
            points = [booms[i - 1], booms[i + 1]]
        else:
            points = booms[i - 1:i + 2]
        out[boom] = (points, log_gradient_weights([HEIGHTS[p] for p in points], HEIGHTS[boom]))
    return out


def _wide(rows: pd.DataFrame, variable: str, stat: str | None, index) -> pd.DataFrame:
    rows = rows[rows["variable"] == variable]
    rows = rows[rows["stat"].isna()] if stat is None else rows[rows["stat"] == stat]
    return rows.pivot_table(index=index, columns="boom", values="value", aggfunc="first", dropna=False)


def _apply(x: np.ndarray, weights: dict, booms: list[int]) -> np.ndarray:
    """Columns (booms) of x -> each boom's weighted sum over its stencil."""
    col = {b: j for j, b in enumerate(booms)}
    out = np.full(x.shape, np.nan)
    for boom, (points, w) in weights.items():
        out[:, col[boom]] = x[:, [col[p] for p in points]] @ w
    return out


def _rows(values: dict[str, np.ndarray], index: pd.Index, booms: list[int], variant: str | None) -> list[pd.DataFrame]:
    frames = []
    for variable, arr in values.items():
        arr = np.where(np.isfinite(arr), arr, np.nan)
        df = pd.DataFrame(arr, index=index, columns=pd.Index(booms, name="boom")).stack(future_stack=True)
        df = df.rename("value").reset_index().assign(variable=variable)
        frames.append(df if variant is None else df.assign(variant=variant))
    return frames


def gradients(boom_final: pd.DataFrame, booms, stencils) -> pd.DataFrame:
    """`boom_final`-shaped rows (stat null): the profile products for every slot
    with `none` rows, the flux products for every (slot, boom, variant) with a u* row.
    """
    booms = sorted(booms)
    rows = boom_final[boom_final["variable"].isin([*NONE_INPUTS, *FLUX_INPUTS])]
    rows = rows.astype({"variant": object, "variable": object, "stat": object})
    none, flux = rows[rows["variant"] == "none"], rows[rows["variant"] != "none"]
    slots = pd.Index(np.unique(none["slot"].to_numpy()), name="slot")
    p = {v: _wide(none, v, s, "slot").reindex(index=slots, columns=booms).to_numpy(float)
         for v, s in NONE_INPUTS.items()}
    keys = _wide(flux, "ustar", None, ["slot", "variant"]).index
    has_flux = isinstance(keys, pd.MultiIndex) and len(keys) > 0
    f = {v: _wide(flux, v, s, ["slot", "variant"]).reindex(index=keys, columns=booms).to_numpy(float)
         for v, s in FLUX_INPUTS.items()} if has_flux else {}
    at = slots.get_indexer(keys.get_level_values("slot")) if has_flux else np.array([], dtype=int)

    def lift(a: np.ndarray) -> np.ndarray:
        """slot-level (none) values at each flux key's slot; NaN where the slot has none."""
        return np.where((at >= 0)[:, None], a[at], np.nan) if a.size else np.full((at.size, len(booms)), np.nan)

    z = np.array([HEIGHTS[b] for b in booms], dtype=float)
    g = np.array([_GRAVITY[b] for b in booms], dtype=float)
    angle = streamwise_angle(p["ue"], p["vn"])
    g_theta = g / p["vpt"]  # g/θv at each boom
    frames = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for form in stencils:
            weights = stencil(booms, form)
            d = {v: _apply(p[v], weights, booms) for v in ("ue", "vn", "pt", "vpt")}
            d["ws"] = _apply(np.hypot(p["ue"], p["vn"]), weights, booms)
            shear = np.hypot(d["ue"], d["vn"])
            n2 = g_theta * d["vpt"]
            profile = {"dpt_dz": d["pt"], "dvpt_dz": d["vpt"], "dws_dz": d["ws"], "shear": shear, "n2": n2,
                       "rig": n2 / shear**2}
            frames += _rows({name(v, form, stencils): a for v, a in profile.items()}, slots, booms, "none")
            if not has_flux:
                continue

            du_s, dv_s = rotate_streamwise(lift(d["ue"]), lift(d["vn"]), lift(angle))
            stress_shear = f["uw"] * du_s + f["vw"] * dv_s  # minus the shear production
            km = -stress_shear / lift(shear) ** 2
            kh = -f["wvpts"] / lift(d["vpt"])
            out = {"phi_m": KAPPA * z * lift(d["ws"]) / f["ustar"],
                   "phi_h": KAPPA * z * lift(d["pt"]) / f["theta_star"],
                   "phi_h_v": KAPPA * z * lift(d["vpt"]) / f["theta_star"],
                   "rif": lift(g_theta) * f["wvpts"] / stress_shear,
                   "km": km, "kh": kh, "prandtl": km / kh,
                   "l_buoyant": f["sigma_w"] / np.sqrt(lift(n2))}
            frames += _rows({name(v, form, stencils): a for v, a in out.items()}, keys, booms, None)
    if not frames:
        return pd.DataFrame(columns=_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["stat"] = None
    return out[_COLUMNS]
