import numpy as np

from ttu_tower.config import config_from_dict
from ttu_tower.math.polar import bearing_to_vector
from ttu_tower.primary.products import file_products
from ttu_tower.primary.stage_a import StageA
from ttu_tower.primary.stage_b import build_span, stage_b
from ttu_tower.timegrid import SAMPLES_PER_HALF_HOUR

N = SAMPLES_PER_HALF_HOUR
_VARS = ("ue", "vn", "w", "ts", "t", "rh", "p")


def _noise(seed, n, std):
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(0.0, 1.0, n), -2.5, 2.5) * std


def _stage_a(seed, ue=5.0, vn=0.0, w=0.0, ts=290.0, t=290.0, rh=0.5, p=90.0, n=N, mutate=None):
    values = {
        "ue": ue + _noise(seed, n, 0.05), "vn": vn + _noise(seed + 1, n, 0.05), "w": w + _noise(seed + 2, n, 0.03),
        "ts": ts + _noise(seed + 3, n, 0.05), "t": t + _noise(seed + 4, n, 0.01),
        "rh": np.clip(rh + _noise(seed + 5, n, 0.002), 0.0, 1.0), "p": p + _noise(seed + 6, n, 0.01),
    }
    if mutate is not None:
        values = mutate(values)
    return StageA(**values, bounds_removed={v: np.zeros(n, dtype=bool) for v in _VARS})


def _cfg(**sections):
    raw = {"tag": "t", "paths": {"raw_dirs": ["/data"]}, "files": {"bad_records": []}}
    raw.update(sections)
    return config_from_dict(raw)


def _run_unit(half_hours, cfg, mutate_by_h=None):
    """Build StageA for `half_hours` and Stage B for their interior, using the
    same rolling-buffer construction stream.run_unit will use.
    """
    mutate_by_h = mutate_by_h or {}
    A = {h: _stage_a(seed=h, mutate=mutate_by_h.get(h)) for h in half_hours}
    B = {}
    for h in half_hours[1:-1]:
        span = build_span(A, h, margin=30_000)
        B[h] = stage_b(span, h, boom=1, cfg=cfg)
    return B


def test_no_file_gives_only_slot_boom_rows():
    cfg = _cfg()
    B = _run_unit(range(200, 207), cfg)
    del B[203]  # h=203 has no file at all
    out = file_products(203, B, boom=1, cfg=cfg)
    assert set(out) == {"slot_boom"}
    assert (out["slot_boom"]["status"] == "no_file").all()
    assert not out["slot_boom"]["unexcised_computed"].any()


def test_computed_slot_has_mrd_and_ladder_rows():
    cfg = _cfg()
    B = _run_unit(range(200, 207), cfg)
    out = file_products(203, B, boom=1, cfg=cfg)

    assert (out["slot_boom"]["status"] == "computed").all()
    assert not out["mrd"].empty
    assert not out["mrd_frame"].empty
    assert not out["ladder"].empty
    assert not out["ladder_coverage"].empty
    assert set(out["mrd"]["variant"]) == {"mrd"}
    # 15 modes (default floor_level) per spectrum, per slot; 3 slots in file 203
    assert out["mrd"].groupby(["slot", "spectrum"]).size().eq(15).all()
    assert out["mrd"]["slot"].nunique() == 3


def test_all_flagged_slot_is_no_data_with_unexcised_counterfactual():
    ue0, vn0 = bearing_to_vector(5.0, 140.0)  # inside the (105, 170) shadow sector -> direction-flagged

    def flag_direction(values):
        values = dict(values)
        values["ue"][:] = ue0 + _noise(999, N, 0.05)
        values["vn"][:] = vn0 + _noise(998, N, 0.05)
        return values

    cfg = _cfg()
    B = _run_unit(range(200, 207), cfg, mutate_by_h={203: flag_direction})
    out = file_products(203, B, boom=1, cfg=cfg)

    slot_boom = out["slot_boom"]
    assert (slot_boom["status"] == "no_data").all()
    assert slot_boom["unexcised_computed"].all()  # the raw (unflagged) data is still there for the counterfactual
    assert set(out["mrd"]["variant"]) == {"mrd_unexcised"}


def test_unexcised_trigger_reaches_neighbouring_slots():
    def inject_gap(values):
        values = dict(values)
        values["ue"][:200] = np.nan  # a gap too long for fill_short, at the very start of h=203's core
        values["vn"][:200] = np.nan
        values["w"][:200] = np.nan
        return values

    cfg = _cfg(qc={"max_fill_gap_samples": 1})
    B = _run_unit(range(200, 207), cfg, mutate_by_h={203: inject_gap})

    out_202 = file_products(202, B, boom=1, cfg=cfg)  # the previous file, within +-4 slots of the gap's slot
    out_205 = file_products(205, B, boom=1, cfg=cfg)  # far enough away to be unaffected

    assert out_202["slot_boom"]["unexcised_computed"].any()
    assert not out_205["slot_boom"]["unexcised_computed"].any()
