"""Against the real full-year run, read-only: opening it, a year of one
quantity, and reprocessing a slot back to its stored means.
"""
import time

import numpy as np
import pyarrow.dataset as pa_dataset
import pytest

from ttu_tower.io.runs import locate_home
from ttu_tower.timegrid import SAMPLES_PER_SLOT
from ttu_tower.viewer import timeline
from ttu_tower.viewer.catalog import build_catalog
from ttu_tower.viewer.fragments import FragmentIndex
from ttu_tower.viewer.reprocess import Reprocessor
from ttu_tower.viewer.run import RunHandle, RunNotFound

pytestmark = pytest.mark.integration

_TAG = "oneyear"
_SLOT = 728857  # 2013-11-09 12:10 -06:00
_BOOM = 8


@pytest.fixture(scope="module")
def real_run():
    mp = pytest.MonkeyPatch()
    mp.delenv("TTU_TOWER_HOME", raising=False)  # the real home, which this module only reads
    try:
        if locate_home() is None:
            pytest.skip("no ttu-tower home on this machine")
        try:
            yield RunHandle.open(_TAG)
        except RunNotFound:
            pytest.skip(f"no '{_TAG}' run registered")
    finally:
        mp.undo()


def test_a_year_of_one_quantity_loads_quickly(real_run):
    index = FragmentIndex(real_run.run_dir)
    q = build_catalog(real_run)["boom_final|ws|mean"]
    t0 = time.perf_counter()
    data = timeline.load(real_run, index, q, "none", tuple(range(1, 11)))
    assert time.perf_counter() - t0 < 3.0
    assert all(np.isfinite(c.y).any() for c in data.curves.values())


def test_reprocessed_slot_matches_its_stored_means(real_run):
    assert real_run.can_reprocess, real_run.config_problems
    index = FragmentIndex(real_run.run_dir)
    f = pa_dataset.field
    means = index.read("means", booms=[_BOOM], half_hours=[_SLOT // 3],
                       filter=(f("slot") == _SLOT) & (f("boom") == _BOOM) & (f("stat") == "mean"))
    win = Reprocessor(real_run).qc_applied(_BOOM, _SLOT * SAMPLES_PER_SLOT, SAMPLES_PER_SLOT)
    checked = 0
    for var, fam in (("ue", "momentum"), ("vn", "momentum"), ("w", "momentum")):
        stored = means.loc[means["variable"] == var, "value"].iloc[0]
        value = win.series[var][win.family_masks[fam]].mean()
        assert value == pytest.approx(stored, rel=1e-12)
        checked += 1
    for var in ("t", "rh", "p"):
        stored = means.loc[means["variable"] == var, "value"].iloc[0]
        x, ok = win.slow_smoothed[var]
        assert x[ok].mean() == pytest.approx(stored, rel=1e-12)
        checked += 1
    assert checked == 6
