"""The configurable cadence (slot length and tau ladder): its geometry, the
config rules that depend on it, the ladder and MRD identities at a
non-default cadence (15-min slots, 9 rungs), and a synthetic run through all
three stages at that cadence.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ttu_tower.config import ConfigError, config_from_dict
from ttu_tower.config.hashing import primary_hash
from ttu_tower.io.runs import find_home
from ttu_tower.io.store import read_table
from ttu_tower.primary.blocks import finest_sums, floor_sums
from ttu_tower.primary.ladder import rung_statistics
from ttu_tower.primary.mrd import detection_outputs, mode_scales_s
from ttu_tower.primary.runner import run_primary
from ttu_tower.secondary.runner import run_secondary
from ttu_tower.tertiary.runner import run_tertiary
from ttu_tower.timegrid import (
    DEFAULT_CADENCE, SAMPLES_PER_HALF_HOUR, SAMPLES_PER_SLOT, BlockGrid, Cadence, block_sums, half_hour_to_time,
)
from ttu_tower.validation.synthetic import write_raw_dataset

C15 = Cadence(slot_min=15, n_rungs=9)


# --- geometry ------------------------------------------------------------------------------------

def test_default_cadence_is_the_original_fixed_geometry():
    c = DEFAULT_CADENCE
    assert c.samples_per_slot == SAMPLES_PER_SLOT and c.slots_per_file == 3
    assert c.rung_seconds == (9.375, 18.75, 37.5, 75.0, 150.0, 300.0, 600.0, 1200.0)
    assert c.finest == BlockGrid.FINEST
    assert all(c.floor(k) == BlockGrid.floor(k) for k in (9, 12, 15, 16))
    assert c.detection_window(7) == (30000 * 7 - 35 * 60 * 50, 80 * 60 * 50)
    assert c.ladder_window(7) == (30000 * 7 - 5 * 60 * 50, 20 * 60 * 50)
    assert c.finest_per_half_hour == 192 and c.finest_per_window == 128
    assert c.slot_rung_s == 600.0 and c.top_rung_s == 1200.0


def test_15_min_nine_rung_geometry():
    c = C15
    assert c.samples_per_slot == 45_000 and c.slots_per_file == 2
    assert c.file_slots(10) == [20, 21] and c.slot_file(21) == 10
    assert c.rung_seconds == (7.03125, 14.0625, 28.125, 56.25, 112.5, 225.0, 450.0, 900.0, 1800.0)
    assert c.finest.length_samples == 351.5625 and c.finest_per_slot == 128
    assert c.detection_window(4) == (45_000 * 4 - 157_500, 360_000)  # [slot - 52.5 min, slot end + 52.5 min)
    assert c.ladder_window(4) == (45_000 * 4 - 22_500, 90_000)  # [slot - 7.5 min, slot end + 7.5 min)
    # the floor grid tiles both the half-hour file and the detection window exactly
    floor = c.floor(15)
    assert c.floor_blocks_per_half_hour(15) * floor.length_samples == SAMPLES_PER_HALF_HOUR
    assert 2**15 * floor.length_samples == c.detection_window_samples
    # every rung is an MRD mode scale, so a statistic at a rung contains exactly the modes up to it
    scales = c.mode_scales_s(15)
    assert all(np.isclose(scales, r).any() for r in c.rung_seconds)
    assert scales[-1] == 2 * 3600.0
    np.testing.assert_allclose(mode_scales_s(15, cadence=c), scales)
    # slots are anchored at the epoch, on quarter hours in UTC
    assert c.time_to_slot(pd.Timestamp("2013-12-09 18:00", tz="UTC")) * 15 % 60 == 0
    assert c.slot_to_time(c.time_to_slot(pd.Timestamp("2013-12-09 18:15", tz="UTC"))) == pd.Timestamp(
        "2013-12-09 18:15", tz="UTC")


def test_cadence_rejects_slots_that_do_not_suit_the_files():
    with pytest.raises(ValueError):
        Cadence(slot_min=12)
    with pytest.raises(ValueError):
        Cadence(slot_min=15, n_rungs=2)


# --- config --------------------------------------------------------------------------------------

def _raw(**sections):
    return {"tag": "t", "paths": {"raw_dirs": ["x"]}, "files": {"bad_records": []}, **sections}


def test_tau_defaults_follow_the_cadence():
    cfg = config_from_dict(_raw(cadence={"slot_min": 15, "n_rungs": 9}))
    assert Cadence.of(cfg) == C15
    det, sel = cfg.secondary.detection, cfg.secondary.selection
    assert (det.min_tau_s, sel.fallback_tau_s, det.max_tau_s) == (7.03125, 900.0, 1800.0)
    default = config_from_dict(_raw())
    assert Cadence.of(default) == DEFAULT_CADENCE
    assert (default.secondary.detection.min_tau_s, default.secondary.selection.fallback_tau_s,
            default.secondary.detection.max_tau_s) == (9.375, 600.0, 1200.0)


@pytest.mark.parametrize("sections, match", [
    ({"cadence": {"slot_min": 20}}, "cadence.slot_min"),
    ({"cadence": {"slot_min": 15}, "secondary": {"detection": {"min_tau_s": 9.375}}}, "ladder rung"),
    ({"cadence": {"slot_min": 15}, "ladder": {"gust_periods_s": [7]}}, "divide the 900 s slot"),
    ({"cadence": {"n_rungs": 10}, "mrd": {"floor_level": 10}}, "floor_level"),
    ({"cadence": {"bogus": 1}}, "unknown key"),
])
def test_cadence_dependent_config_rules(sections, match):
    with pytest.raises(ConfigError, match=match):
        config_from_dict(_raw(**sections))


def test_default_cadence_leaves_the_primary_hash_alone():
    # A config without [cadence] and one spelling out the default hash the same,
    # so runs made before [cadence] existed still pass their hash guard.
    plain = config_from_dict(_raw())
    explicit = config_from_dict(_raw(cadence={"slot_min": 10, "n_rungs": 8}))
    other = config_from_dict(_raw(cadence={"slot_min": 15, "n_rungs": 9}))
    assert primary_hash(plain, "2.0.0") == primary_hash(explicit, "2.0.0")
    assert primary_hash(other, "2.0.0") != primary_hash(plain, "2.0.0")


# --- ladder and MRD identities at 15 min ---------------------------------------------------------

def _window(rng, c: Cadence, k: int):
    g0, n = c.ladder_window(k)
    series = {"ue": rng.normal(3, 1, n), "vn": rng.normal(1, 1, n), "w": rng.normal(0, 0.5, n),
              "ts": rng.normal(290, 1, n), "vpts": rng.normal(295, 1, n)}
    masks = {"momentum": rng.random(n) > 0.05, "heat": rng.random(n) > 0.05, "ts": rng.random(n) > 0.05}
    return g0, series, masks


def test_ladder_at_15_min_matches_blocks_of_each_rung_directly():
    """Each rung's w variance and w'vpts' covariance, aggregated from finest
    sums, equal those computed on a grid of that rung's own block length:
    the blocks tile the slot below the top rung, and the top rung is the one
    2-slot window centered on it.
    """
    c, k, cov_c = C15, 3, 0.75
    rng = np.random.default_rng(5)
    g0, series, masks = _window(rng, c, k)
    b0 = (c.samples_per_slot * k - c.ladder_margin_samples) * c.finest.den // c.finest.num
    finest = finest_sums(series, masks, g0, b0, c.finest_per_window, c)
    values, coverage = rung_statistics(finest, cov_c, c)
    got = {(r, v, s): x for r, v, s, x in values}
    totals = {(r, fam): total for r, fam, _, total, _ in coverage}

    w, theta = series["w"], series["vpts"] - 273.15
    for j, rung in enumerate(c.rung_seconds):
        if j < c.n_rungs - 1:
            per_slot = c.finest_per_slot // 2**j
            grid = BlockGrid(c.samples_per_slot, per_slot)
            start, nb = c.samples_per_slot * k * per_slot // c.samples_per_slot, per_slot
        else:
            grid, start, nb = BlockGrid(c.ladder_window_samples, 1), None, 1
        if start is None:  # the top rung: the whole window [g0, g0 + 2 S) is its one block
            sx, n = _whole(np.stack([w, w**2], 1), masks["momentum"])
            sh, nh = _whole(np.stack([w, theta, w * theta], 1), masks["heat"])
        else:
            sx, n = block_sums(np.stack([w, w**2], 1), masks["momentum"].astype(float), g0, grid, start, nb)
            sh, nh = block_sums(np.stack([w, theta, w * theta], 1), masks["heat"].astype(float), g0, grid, start, nb)
        length = grid.length_samples
        ok = n / length >= cov_c
        var_w = np.maximum(sx[:, 1] / n - (sx[:, 0] / n) ** 2, 0.0)
        ok_h = nh / length >= cov_c
        cov_wt = sh[:, 2] / nh - (sh[:, 0] / nh) * (sh[:, 1] / nh)
        np.testing.assert_allclose(got[(rung, "w", "var")], var_w[ok].mean(), rtol=1e-9)
        np.testing.assert_allclose(got[(rung, "wvpts", "cov")], cov_wt[ok_h].mean(), rtol=1e-9, atol=1e-12)
        assert totals[(rung, "momentum")] == nb


def _whole(values, mask):
    """Sums over the whole top-rung window (a single block)."""
    sx = np.where(mask[:, None], values, 0.0).sum(axis=0, keepdims=True)
    return sx, np.array([float(mask.sum())])


def test_mrd_modes_sum_to_the_within_block_covariance_at_15_min():
    """The identity behind tau: the w'theta' modes with P <= tau sum to the
    mean within-block covariance (of floor-block means) over blocks of tau,
    for every rung of the 15-min ladder.
    """
    c, level = C15, 12
    rng = np.random.default_rng(9)
    n = c.detection_window_samples
    series = {"ue": rng.normal(3, 1, n), "vn": rng.normal(1, 1, n), "w": rng.normal(0, 0.5, n),
              "ts": rng.normal(290, 1, n), "vpts": rng.normal(295, 1, n) + np.linspace(0, 2, n)}
    masks = {fam: np.ones(n, bool) for fam in ("momentum", "heat", "ts")}
    floor = floor_sums(series, masks, 0, level, 0, 2**level, c)
    spectra, _, _, cov_heat = detection_outputs(floor, c=0.5, cadence=c)
    assert cov_heat == pytest.approx(1.0)
    value = spectra["wvpts"][0]
    scales = c.mode_scales_s(level)

    heat = floor["heat"]
    w_f = heat.sx[:, heat.columns.index("w")] / heat.n
    t_f = heat.sx[:, heat.columns.index("theta")] / heat.n
    for rung in c.rung_seconds:
        per_block = int(round(rung / (c.floor(level).length_samples / 50)))
        wb, tb = w_f.reshape(-1, per_block), t_f.reshape(-1, per_block)
        within = ((wb - wb.mean(1, keepdims=True)) * (tb - tb.mean(1, keepdims=True))).mean(1).mean()
        np.testing.assert_allclose(value[scales <= rung + 1e-9].sum(), within, rtol=1e-9, atol=1e-14)


# --- all three stages at 15 min ------------------------------------------------------------------

_HALF_HOURS = list(range(6300, 6310))
_BOOMS = [1, 2]


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("cadence15") / "raw"
    write_raw_dataset(d, _HALF_HOURS, _BOOMS, np.random.default_rng(21), missing=[6304])
    return d


def test_pipeline_at_15_min_slots(tmp_path, monkeypatch, raw_dir):
    monkeypatch.setenv("TTU_TOWER_HOME", str(tmp_path / "home"))
    start = half_hour_to_time(_HALF_HOURS[0] + 1).tz_convert("UTC")
    end = half_hour_to_time(_HALF_HOURS[-1]).tz_convert("UTC")
    cfg = config_from_dict({
        "tag": "cad15", "paths": {"raw_dirs": [str(raw_dir)]},
        "output": {"timezone": "UTC"},
        "period": {"start": start.strftime("%Y-%m-%d %H:%M"), "end": end.strftime("%Y-%m-%d %H:%M")},
        "cadence": {"slot_min": 15, "n_rungs": 9},
        "files": {"bad_records": [], "booms": _BOOMS},
        "primary": {"batch_max_files": 3},
        "tertiary": {"veer_reference_boom": 1, "fits": {k: _BOOMS for k in
                     ("alpha_booms", "gamma_booms", "wdgamma_booms", "loglaw_booms")}},
    })
    args = SimpleNamespace(config=str(tmp_path / "cad15.toml"), nproc=1, test=False, redo_failures=False,
                           force=False, allow_non_parquet=False)
    run_primary(cfg, args)
    run_secondary(cfg, args)
    run_tertiary(cfg, args)
    run_dir = find_home() / "results" / "cad15"
    c = C15

    slots = pd.read_parquet(run_dir / "primary" / "slots.parquet").sort_values("slot")
    expected = [k for h in range(_HALF_HOURS[0] + 1, _HALF_HOURS[-1]) for k in c.file_slots(h)]
    assert slots["slot"].tolist() == expected
    assert (slots["slot_start"].diff().dropna() == pd.Timedelta(minutes=15)).all()
    assert slots["slot_start"].iloc[0] == start

    ladder = read_table(run_dir / "primary" / "data" / "ladder")
    assert sorted(ladder["rung_s"].unique()) == list(c.rung_seconds)
    lcov = read_table(run_dir / "primary" / "data" / "ladder_coverage")
    lcov = lcov[lcov["family"] == "momentum"].groupby("rung_s")["blocks_total"].first()
    assert lcov.to_dict() == {r: (c.finest_per_slot // 2**j if j < 8 else 1) for j, r in enumerate(c.rung_seconds)}
    mrd = read_table(run_dir / "primary" / "data" / "mrd")
    np.testing.assert_allclose(np.sort(mrd["scale_s"].unique()), c.mode_scales_s(15))

    tau_sel = read_table(run_dir / "secondary" / "data" / "tau_selected")
    assert set(tau_sel.loc[tau_sel["variant"] == "naive", "tau_s"]) == {900.0}
    assert set(tau_sel["tau_s"].dropna()) <= set(c.rung_seconds)
    boom_stats = read_table(run_dir / "secondary" / "data" / "boom_stats")
    assert boom_stats.loc[boom_stats["variant"] == "naive", "value"].notna().any()

    slot_final = read_table(run_dir / "tertiary" / "data" / "slot_final")
    assert set(slot_final["slot"]) == set(expected)
    wide = pd.concat([pd.read_parquet(p) for p in sorted((run_dir / "tertiary" / "wide").glob("*.parquet"))])
    wide_times = pd.DatetimeIndex(wide["slot_start"] if "slot_start" in wide.columns else wide.index).sort_values()
    assert (wide_times.to_series().diff().dropna() == pd.Timedelta(minutes=15)).all()
    assert len(wide_times) == len(expected)

    # the viewer reads the cadence back from the run
    from ttu_tower.viewer.run import RunHandle
    assert RunHandle.from_dir(run_dir).cadence == c


def test_viewer_follows_the_open_runs_cadence():
    from ttu_tower.viewer import reprocess, slotqc, taustats, tauvariants, timeaxis, whatif

    timeaxis.set_cadence(C15)
    try:
        assert timeaxis.slot_seconds() == 900.0 and timeaxis.SLOT_SECONDS == 900.0
        assert timeaxis.unix_to_slot(timeaxis.slot_to_unix(12345) + 899) == 12345
        assert timeaxis.parse_time("2013-12-09 18:29", "UTC") == pd.Timestamp("2013-12-09 18:15", tz="UTC")
        assert "18:15–18:30" in timeaxis.format_slot(C15.time_to_slot(pd.Timestamp("2013-12-09 18:15", tz="UTC")), "UTC")
        assert taustats.rungs() == C15.rung_seconds == taustats.RUNGS
        assert "fixed_900" in tauvariants.all_fixed() and "fixed_900" not in tauvariants.fixed()
        assert whatif.ladder_rungs() == C15.rung_seconds
        assert set(slotqc.windows(3)) == {"slot", "ladder window (30 min)", "detection window (120 min)"}
        assert slotqc.windows(3)["detection window (120 min)"] == (45_000 * 3 - 157_500, 45_000 * 3 + 202_500)
        ids = reprocess.tau_block_ids(0, 45_000, 7.03125)
        assert ids.min() == 0 and ids.max() == 127
        assert reprocess.tau_block_ids(0, 45_000, 900.0).max() == 0
        assert reprocess.tau_block_ids(0, 100, 1800.0) is None
    finally:
        timeaxis.set_cadence(DEFAULT_CADENCE)
    assert timeaxis.slot_seconds() == 600.0 and taustats.RUNGS[-1] == 1200.0
