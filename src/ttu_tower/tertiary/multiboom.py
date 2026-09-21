"""Multi-boom quantities (`pairs`): bulk Richardson number and VPT lapse rate
for every boom pair, and veer relative to a reference boom - all read from
already-filtered `variant="none"` values, so a filtered boom's NaN
propagates automatically.
"""
import itertools

import pandas as pd

from ttu_tower.constants import HEIGHTS, SITE_ELEVATION, SITE_LATITUDE
from ttu_tower.math.polar import series_signed_angular_distance
from ttu_tower.physics.earth import local_gravity
from ttu_tower.physics.richardson import bulk_richardson_number

_PAIRS_COLUMNS = ["slot", "boom", "boom2", "variant", "variable", "value"]


def _wide_none(boom_final: pd.DataFrame, variable: str, stat: str | None = None) -> pd.DataFrame:
    """slot x boom, for a `variant="none"` value (mean-stat `means` rows
    need `stat`; `slow` rows, with no `stat` column, pass `stat=None`).
    """
    mask = (boom_final["variant"] == "none") & (boom_final["variable"] == variable)
    mask &= boom_final["stat"].isna() if stat is None else boom_final["stat"] == stat
    series = boom_final[mask].set_index(["slot", "boom"])["value"]
    return series.unstack("boom")


def multiboom(boom_final: pd.DataFrame, booms: list[int], veer_reference_boom: int) -> pd.DataFrame:
    booms_sorted = sorted(booms)
    ue = _wide_none(boom_final, "ue", "mean").reindex(columns=booms_sorted)
    vn = _wide_none(boom_final, "vn", "mean").reindex(columns=booms_sorted)
    vpt = _wide_none(boom_final, "vpt").reindex(columns=booms_sorted)
    wd = _wide_none(boom_final, "wd", "mean").reindex(columns=booms_sorted)

    slots = ue.index.union(vn.index).union(vpt.index).union(wd.index)
    ue, vn, vpt, wd = (df.reindex(slots) for df in (ue, vn, vpt, wd))

    rows = []
    for b1, b2 in itertools.combinations(booms_sorted, 2):
        gravity = local_gravity(SITE_LATITUDE, SITE_ELEVATION + (HEIGHTS[b1] + HEIGHTS[b2]) / 2)
        rib = bulk_richardson_number(
            vpt[b1].to_numpy(), vpt[b2].to_numpy(), HEIGHTS[b1], HEIGHTS[b2],
            ue[b1].to_numpy(), ue[b2].to_numpy(), vn[b1].to_numpy(), vn[b2].to_numpy(),
            components=True, gravity=gravity,
        )
        lapse = (vpt[b2].to_numpy() - vpt[b1].to_numpy()) / (HEIGHTS[b2] - HEIGHTS[b1])
        for slot, rib_v, lapse_v in zip(slots, rib, lapse):
            rows.append((slot, b1, b2, "rib", float(rib_v)))
            rows.append((slot, b1, b2, "lapse_vpt", float(lapse_v)))

    # boom == boom2 for the reference boom's own row is deliberate: veer uses
    # boom2 as the reference, not as a pair partner, unlike rib/lapse_vpt's
    # boom < boom2 pairing. Every row (including the reference's own) goes
    # through the same call, so a filtered reference boom gives NaN veer
    # everywhere rather than a hardcoded 0.
    reference = wd[veer_reference_boom].to_numpy()
    for boom in booms_sorted:
        veer = series_signed_angular_distance(wd[boom].to_numpy(), reference)
        for slot, v in zip(slots, veer):
            rows.append((slot, boom, veer_reference_boom, "veer", float(v)))

    df = pd.DataFrame(rows, columns=["slot", "boom", "boom2", "variable", "value"])
    df["variant"] = "none"
    return df[_PAIRS_COLUMNS]
