import dataclasses

import numpy as np
import pytest

from ttu_tower.flags import FlagStore
from ttu_tower.io.store import read_table
from ttu_tower.timegrid import SAMPLES_PER_HALF_HOUR, SAMPLES_PER_SLOT
from ttu_tower.viewer import qcwhatif
from ttu_tower.viewer.fragments import FragmentIndex, alias_bounds_variables
from ttu_tower.viewer.reprocess import Reprocessor
from ttu_tower.viewer.run import RunHandle

from viewer_fixtures import FAULT_HALF_HOUR, SPIKY_BOOM, SPIKY_SLOT, TS_SPIKE_INDEX


@pytest.fixture(scope="module")
def primary(primary_run):
    run = RunHandle.from_dir(primary_run.run_dir)
    return run, FragmentIndex(run.run_dir), Reprocessor(run)


def test_fields_cover_every_qc_setting_and_edit_one_at_a_time(primary):
    run, _, _ = primary
    fields = qcwhatif.fields(run.cfg.qc)
    names = {f.name for f in fields}
    assert {"min_coverage", "unusable_tests", "bounds.sonic", "despike.z_threshold.ts", "despike.min_mad.rh",
            "windows.skew_range", "second_layer.shadow_sector", "slow.smoothing_width_s.p"} <= names
    edited = qcwhatif.with_values(run.cfg, {("despike", "z_threshold", "ts"): 9.0, ("min_coverage",): 0.5})
    assert edited.qc.despike.z_threshold["ts"] == 9.0 and edited.qc.min_coverage == 0.5
    assert edited.qc.despike.z_threshold["ue"] == run.cfg.qc.despike.z_threshold["ue"]
    assert run.cfg.qc.despike.z_threshold["ts"] != 9.0  # the run's config is untouched
    assert {c[0] for c in qcwhatif.changes(run.cfg.qc, edited.qc)} == {"despike.z_threshold.ts", "min_coverage"}
    assert qcwhatif.problems(edited) == []
    assert qcwhatif.problems(qcwhatif.with_values(run.cfg, {("despike", "window_s"): 5000.0}))


def test_unchanged_parameters_reproduce_the_stored_flags_and_means(primary):
    run, index, base = primary
    rp = qcwhatif.reprocessor(base, dataclasses.replace(run.cfg))
    h = FAULT_HALF_HOUR
    slots = range(3 * h, 3 * h + 3)
    table = qcwhatif.slot_comparison(index, rp, 1, slots)
    assert len(table) > 20 and table["run"].notna().sum() > 20
    assert np.array_equal(table["run"].to_numpy(), table["what-if"].to_numpy(), equal_nan=True)
    stored = FlagStore(alias_bounds_variables(read_table(run.run_dir / "primary" / "data" / "flags")))
    fresh = FlagStore(alias_bounds_variables(rp.stage_b(1, h)[0].flags))
    g0 = 3 * h * SAMPLES_PER_SLOT
    for test in ("bounds", "spike", "unchecked", "resolution", "dropouts", "skew", "kurt"):
        for var in ("ue", "vn", "w", "ts", "t", "rh", "p"):
            assert np.array_equal(stored.mask([test], 1, var, g0, SAMPLES_PER_HALF_HOUR),
                                  fresh.mask([test], 1, var, g0, SAMPLES_PER_HALF_HOUR)), (test, var)


def test_a_higher_threshold_keeps_the_injected_spike(primary):
    run, _, base = primary
    rp = qcwhatif.reprocessor(base, qcwhatif.with_values(run.cfg, {("despike", "z_threshold", "ts"): 1e6}))
    g = FAULT_HALF_HOUR * SAMPLES_PER_HALF_HOUR + TS_SPIKE_INDEX
    g0, n = g - 1000, 2000
    assert base.qc_applied(1, g0, n).removed["spike"]["ts"][1000]
    assert not rp.qc_applied(1, g0, n).removed.get("spike", {}).get("ts", np.zeros(n, bool))[1000]


def test_whatif_caches_never_touch_the_runs_own(primary):
    run, _, base = primary
    h = FAULT_HALF_HOUR
    stored_b = base.stage_b(1, h)
    before = set(base._B._data)
    rp = qcwhatif.reprocessor(base, qcwhatif.with_values(run.cfg, {("min_coverage",): 0.5}))
    assert rp._stage_a_from is base  # same bounds: Stage A is shared
    out, _ = rp.stage_b(1, h)
    assert out is not stored_b[0] and base.stage_b(1, h) is stored_b
    assert set(base._B._data) == before
    bounded = qcwhatif.reprocessor(base, qcwhatif.with_values(run.cfg, {("bounds", "ts"): (270.0, 300.0)}))
    assert bounded._stage_a_from is None
    assert bounded.stage_a(1, h) is not base.stage_a(1, h)


@pytest.mark.slow
def test_slot_tau_with_the_runs_qc_reproduces_the_stored_tau(full_run):
    run = RunHandle.from_dir(full_run.run_dir)
    index = FragmentIndex(run.run_dir)
    base = Reprocessor(run)
    same = qcwhatif.reprocessor(base, dataclasses.replace(run.cfg))
    table = qcwhatif.slot_tau(index, same, SPIKY_BOOM, SPIKY_SLOT - 3, run.cfg.secondary)
    assert set(table["variant"]) >= {"mrd"}
    for side in ("heat", "momentum", "source", "u*", "σw"):
        a, b = table[f"{side} (run)"].to_numpy(), table[f"{side} (what-if)"].to_numpy()
        assert (a == b).all() or np.allclose(a.astype(float), b.astype(float), equal_nan=True), side
    assert np.array_equal(table["τ (run)"].to_numpy(), table["τ (what-if)"].to_numpy(), equal_nan=True)
