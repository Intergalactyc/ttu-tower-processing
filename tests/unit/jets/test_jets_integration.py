"""Against the real full-year run, read-only: the vectorized detection agrees
with a slot-by-slot reading of the rules on every slot, and the events obey
their own rules.
"""
import time

import numpy as np
import pytest

from ttu_tower.constants import HEIGHTS
from ttu_tower.io.runs import locate_home
from ttu_tower.jets import data, identify
from ttu_tower.jets.events import EventParams, find_events
from ttu_tower.jets.filters import JetFilters, apply_filters, context_needs, event_quantities
from ttu_tower.jets.identify import NoseParams
from ttu_tower.viewer.run import RunHandle, RunNotFound

from jets_reference import reference_nose

pytestmark = pytest.mark.integration

_TAG = "oneyear"


@pytest.fixture(scope="module")
def real_profiles():
    mp = pytest.MonkeyPatch()
    mp.delenv("TTU_TOWER_HOME", raising=False)  # the real home, which this module only reads
    try:
        if locate_home() is None:
            pytest.skip("no ttu-tower home on this machine")
        try:
            run = RunHandle.open(_TAG)
        except RunNotFound:
            pytest.skip(f"no '{_TAG}' run registered")
        t0 = time.perf_counter()
        profiles = data.load_profiles(run.run_dir)
        yield run, profiles, time.perf_counter() - t0
    finally:
        mp.undo()


def test_a_year_loads_and_detects_quickly(real_profiles):
    _, profiles, load_s = real_profiles
    assert load_s < 5.0
    t0 = time.perf_counter()
    identify.detect("nose", profiles)
    assert time.perf_counter() - t0 < 0.5


@pytest.mark.parametrize("params", [NoseParams(), NoseParams(min_nose_speed=8.0, min_falloff_rel=0.2)])
def test_detection_matches_the_rules_on_every_slot(real_profiles, params):
    _, profiles, _ = real_profiles
    out = identify.detect("nose", profiles, params)
    assert out["is_jet"].sum() > 0
    status = out["status"].astype(str).to_numpy()
    booms = out["nose_boom"].to_numpy(dtype=float, na_value=np.nan)
    for i, row in enumerate(profiles.ws):
        want, boom = reference_nose(row, params)
        assert status[i] == want, profiles.slots[i]
        assert (boom is None and np.isnan(booms[i])) or booms[i] == boom, profiles.slots[i]
    assert (out.loc[out["is_jet"], "z_nose"] < HEIGHTS[10]).all()


def test_events_obey_their_rules(real_profiles):
    run, profiles, _ = real_profiles
    detection = identify.detect("nose", profiles)
    events = find_events(detection, EventParams(), profiles)
    table = events.table
    assert len(table) > 0 and (table["n_slots"] >= 6).all()
    assert (table["n_detected"] + table["n_filled"] == table["n_slots"]).all()
    in_event = events.slots["event_id"].notna()
    assert not (detection.loc[in_event, "status"] == "no_data").any()
    assert table["wd_nose"].between(0, 360).all()
    filters = JetFilters(night_only=True, min_inversion=1.0)
    context = data.load_context(run.run_dir, *context_needs(filters))
    passed = apply_filters(event_quantities(events, detection, context, filters), filters)["passed"]
    assert 0 < passed.sum() < len(table)
