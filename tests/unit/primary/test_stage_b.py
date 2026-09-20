import numpy as np
import pytest

from ttu_tower.config import config_from_dict
from ttu_tower.primary.stage_a import StageA
from ttu_tower.primary.stage_b import build_span, stage_b
from ttu_tower.timegrid import SAMPLES_PER_HALF_HOUR

N = SAMPLES_PER_HALF_HOUR
_VARS = ("ue", "vn", "w", "ts", "t", "rh", "p")


def _noise(seed, n, std):
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(0.0, 1.0, n), -2.5, 2.5) * std


def _stage_a(seed=0, ue=5.0, vn=0.0, w=0.0, ts=290.0, t=290.0, rh=0.5, p=90.0, n=N, mutate=None):
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


# --- placeholder -----------------------------------------------------------------

def test_placeholder_for_missing_half_hour():
    cfg = _cfg()
    span = build_span({}, h=100, margin=30_000)
    out = stage_b(span, h=100, boom=1, cfg=cfg)
    assert out.coverage.empty
    assert out.means.empty
    assert out.flags.empty
    assert out.mask_differs == {300: False, 301: False, 302: False}


# --- happy path --------------------------------------------------------------------

def test_happy_path_full_coverage_no_flags():
    cfg = _cfg()
    a99, a100, a101 = _stage_a(seed=0), _stage_a(seed=100), _stage_a(seed=200)
    span = build_span({99: a99, 100: a100, 101: a101}, h=100, margin=30_000)
    out = stage_b(span, h=100, boom=1, cfg=cfg)

    assert out.flags.empty, out.flags
    usable = out.coverage[(out.coverage["variable"] == "ue") & (out.coverage["layer"] == "usable")]
    assert (usable["fraction"] > 0.999).all()
    assert all(v is False for v in out.mask_differs.values())
    assert out.floor["momentum"].n.size == 12288
    assert out.finest["momentum"].n.size == 192

    ue_mean = out.means[(out.means["variable"] == "ue") & (out.means["stat"] == "mean")]
    assert ue_mean["value"].apply(lambda v: v == pytest.approx(5.0, abs=0.01)).all()


def test_placeholder_neighbour_still_processes_center_file():
    cfg = _cfg()
    a100 = _stage_a(seed=100)
    span = build_span({100: a100}, h=100, margin=30_000)  # 99 and 101 missing
    out = stage_b(span, h=100, boom=1, cfg=cfg)
    assert not out.coverage.empty
    usable = out.coverage[(out.coverage["variable"] == "ue") & (out.coverage["layer"] == "usable")]
    # edges of the core lose despike/resolution context, but the bulk is unaffected
    assert usable["fraction"].median() > 0.99


# --- slow-sensor (t, rh, p) smoothing mask ------------------------------------------

def test_slow_sensor_short_gap_is_filled_before_smoothing():
    def inject(values):
        values = dict(values)
        values["p"][14_980:15_020] = np.nan  # a 0.8 s gap, well under max_fill_gap_samples (50)
        return values

    cfg = _cfg()
    a100 = _stage_a(seed=100, mutate=inject)
    span = build_span({99: _stage_a(seed=0), 100: a100, 101: _stage_a(seed=200)}, h=100, margin=30_000)
    out = stage_b(span, h=100, boom=1, cfg=cfg)

    p_present = out.coverage[(out.coverage["variable"] == "p") & (out.coverage["layer"] == "present") & (out.coverage["slot"] == 300)]
    p_usable = out.coverage[(out.coverage["variable"] == "p") & (out.coverage["layer"] == "usable") & (out.coverage["slot"] == 300)]
    assert p_present["fraction"].iloc[0] < 1.0  # the gap shows up as missing raw data
    assert p_usable["fraction"].iloc[0] > 0.999  # but fill_short recovers it before smoothing sees it


def test_slow_sensor_unchecked_sample_excluded_from_usable(monkeypatch):
    # despike()'s own fallback (nearest valid reference within half a window)
    # is forgiving enough that natural fixtures rarely produce a genuinely
    # "unchecked" *finite* sample for p - patch despike() directly so the
    # wiring (first_layer excluding it, downstream of despike) is exercised
    # without depending on tuning despike's own internals to trigger it.
    import ttu_tower.primary.stage_b as stage_b_mod
    from ttu_tower.primary.despike import Despiked, despike as real_despike

    cfg = _cfg()
    a100 = _stage_a(seed=100)
    span = build_span({99: _stage_a(seed=0), 100: a100, 101: _stage_a(seed=200)}, h=100, margin=30_000)

    # core-local [40000, 40100) (slot 301) is span-local [70000, 70100) with a 30000 margin.
    def fake_despike(x, g0, cfg_despike, min_mad, c):
        result = real_despike(x, g0, cfg_despike, min_mad, c)
        if x is span.series["p"]:
            unchecked = result.unchecked.copy()
            unchecked[70_000:70_100] = True  # a finite region forced "unchecked"
            result = Despiked(x=result.x, spike=result.spike, excursion=result.excursion, unchecked=unchecked)
        return result

    monkeypatch.setattr(stage_b_mod, "despike", fake_despike)
    out = stage_b(span, h=100, boom=1, cfg=cfg)

    assert np.isfinite(a100.p[40_000:40_100]).all()  # the forced-unchecked region has real data
    unchecked_p = out.flags[(out.flags["test"] == "unchecked") & (out.flags["variable"] == "p")]
    assert not unchecked_p.empty

    slot = out.coverage[out.coverage["slot"] == 301]  # core samples 30000..60000 -> covers 40000..40100
    p_usable = slot[(slot["variable"] == "p") & (slot["layer"] == "usable")]["fraction"].iloc[0]
    p_present = slot[(slot["variable"] == "p") & (slot["layer"] == "present")]["fraction"].iloc[0]
    assert p_usable < p_present


# --- triplet coupling --------------------------------------------------------------

def test_spike_in_one_wind_component_couples_to_all_three():
    def inject(values):
        values = dict(values)
        values["vn"][45_000:45_003] = 500.0  # a huge spike, well past bounds/despike thresholds
        return values

    # A correctly-classified spike (<= max_spike_samples) is always short
    # enough to be re-filled by fill_short afterward (max_fill_gap_samples
    # default 50 > max_spike_samples default 3) - shrink the fill gap here so
    # the coupled removal is still visible in "filled" coverage.
    cfg = _cfg(qc={"max_fill_gap_samples": 1})
    a100 = _stage_a(seed=100, mutate=inject)
    span = build_span({99: _stage_a(seed=0), 100: a100, 101: _stage_a(seed=200)}, h=100, margin=30_000)
    out = stage_b(span, h=100, boom=1, cfg=cfg)

    ue_filled = out.coverage[(out.coverage["variable"] == "ue") & (out.coverage["layer"] == "filled")]
    # the spike is in slot 301 (samples 30000..60000 of the core -> global 45000)
    slot = ue_filled[ue_filled["slot"] == 301]
    assert slot["fraction"].iloc[0] < 1.0  # ue lost samples too, via triplet coupling
    assert (out.flags["test"] == "spike").any()
    spike_rows = out.flags[(out.flags["test"] == "spike") & (out.flags["variable"] == "vn")]
    assert not spike_rows.empty


# --- direction/bounce final-mask exclusion -----------------------------------------

def test_direction_flag_excludes_slot_from_final_mask():
    from ttu_tower.math.polar import bearing_to_vector

    ue0, vn0 = bearing_to_vector(5.0, 140.0)  # 140 deg FROM-bearing, inside the (105, 170) shadow sector
    a100 = _stage_a(seed=100, ue=ue0, vn=vn0)

    cfg = _cfg()
    span = build_span({99: _stage_a(seed=0), 100: a100, 101: _stage_a(seed=200)}, h=100, margin=30_000)
    out = stage_b(span, h=100, boom=1, cfg=cfg)

    assert (out.flags["test"] == "direction").any()
    usable_l1 = out.coverage[(out.coverage["variable"] == "ue") & (out.coverage["layer"] == "usable_l1")]
    usable = out.coverage[(out.coverage["variable"] == "ue") & (out.coverage["layer"] == "usable")]
    merged = usable_l1.merge(usable, on="slot", suffixes=("_l1", "_final"))
    assert (merged["fraction_final"] < merged["fraction_l1"]).any()


# --- unexcised divergence -----------------------------------------------------------

def test_unexcised_recovers_short_excised_gap_mask_differs():
    def inject(values):
        values = dict(values)
        values["ue"][10_000:10_200] = np.nan  # a 4 s gap: too long for max_fill_gap_samples (default 50)
        values["vn"][10_000:10_200] = np.nan
        values["w"][10_000:10_200] = np.nan
        return values

    cfg = _cfg()
    a100 = _stage_a(seed=100, mutate=inject)
    span = build_span({99: _stage_a(seed=0), 100: a100, 101: _stage_a(seed=200)}, h=100, margin=30_000)
    out = stage_b(span, h=100, boom=1, cfg=cfg)

    assert out.mask_differs[300] is True  # slot 300 = the first slot of file 100 (samples 0..30000)
    assert out.mask_differs[301] is False
    assert out.mask_differs[302] is False


# --- flags are clipped to the core --------------------------------------------------

def test_flags_are_clipped_to_the_core():
    def inject(values):
        values = dict(values)
        values["ue"][-100:] = 1000.0  # a spike right at the end of file 99, bleeding into the margin
        return values

    cfg = _cfg()
    a99 = _stage_a(seed=0, mutate=inject)
    span = build_span({99: a99, 100: _stage_a(seed=100), 101: _stage_a(seed=200)}, h=100, margin=30_000)
    out = stage_b(span, h=100, boom=1, cfg=cfg)

    core_g0 = SAMPLES_PER_HALF_HOUR * 100
    assert (out.flags["start"] >= core_g0).all()
    assert (out.flags["end"] <= core_g0 + SAMPLES_PER_HALF_HOUR).all()
