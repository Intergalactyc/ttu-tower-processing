import numpy as np
import pandas as pd
import pyarrow.dataset as pa_dataset
import pytest

from ttu_tower.flags import FlagStore
from ttu_tower.io.store import read_table
from ttu_tower.timegrid import SAMPLES_PER_SLOT, slot_to_time
from ttu_tower.viewer import timeline
from ttu_tower.viewer.catalog import build_catalog
from ttu_tower.viewer.fragments import FragmentIndex, alias_bounds_variables
from ttu_tower.viewer.run import RunHandle


@pytest.fixture(scope="module")
def primary_opened(primary_run):
    run = RunHandle.from_dir(primary_run.run_dir)
    return run, FragmentIndex(run.run_dir)


def _q(run, key):
    return build_catalog(run)[key]


def test_coverage_curve_equals_a_direct_read(primary_run, primary_opened):
    run, index = primary_opened
    q = _q(run, "coverage|momentum|usable")
    curve = timeline.load(run, index, q, "none", [1]).curves[1]
    f = pa_dataset.field
    direct = read_table(primary_run.run_dir / "primary" / "data" / "coverage",
                        filter=(f("variable") == "momentum") & (f("layer") == "usable") & (f("boom") == 1))
    direct = direct.sort_values("slot")
    np.testing.assert_array_equal(curve.slots, direct["slot"].to_numpy())
    np.testing.assert_array_equal(curve.y, direct["fraction"].to_numpy())


def test_x_is_the_slot_start(primary_opened):
    run, index = primary_opened
    curve = timeline.load(run, index, _q(run, "slot_qc|ts|skew"), "none", [1]).curves[1]
    expected = slot_to_time(curve.slots).tz_convert("UTC")
    got = pd.to_datetime(curve.x, unit="s", utc=True)
    assert (got == expected).all()


def test_flag_fractions_match_the_flag_store(primary_run, primary_opened):
    run, index = primary_opened
    curve = timeline.load(run, index, _q(run, "flags|spike|ts"), "none", [1]).curves[1]
    frame = alias_bounds_variables(read_table(primary_run.run_dir / "primary" / "data" / "flags"))
    store = FlagStore(frame)
    starts = curve.slots * SAMPLES_PER_SLOT
    np.testing.assert_allclose(curve.y, store.fraction("spike", 1, "ts", starts, starts + SAMPLES_PER_SLOT))
    assert (curve.y > 0).any()


def test_categorical_status_codes(primary_opened):
    run, index = primary_opened
    data = timeline.load(run, index, _q(run, "slot_boom|status"), "none", [1])
    assert data.categories == list(timeline.SLOT_STATUSES)
    codes = data.curves[1].y
    assert set(np.unique(codes[np.isfinite(codes)]).astype(int)) <= {0, 1, 2}
    assert (codes == 0).any()  # the missing file's slots are no_file


def test_availability_image_matches_slot_boom(primary_run, primary_opened):
    run, index = primary_opened
    slots, image = timeline.availability(run, index, [1])
    slot_boom = read_table(primary_run.run_dir / "primary" / "data" / "slot_boom")
    by_slot = dict(zip(slot_boom["slot"], slot_boom["status"].astype(str)))
    for j, k in enumerate(slots):
        assert timeline.SLOT_STATUSES[image[0, j]] == by_slot[k]


@pytest.mark.slow
def test_boom_final_curve_and_overlays(full_run):
    run = RunHandle.from_dir(full_run.run_dir)
    index = FragmentIndex(run.run_dir)
    q = build_catalog(run)["boom_final|ws|mean"]
    data = timeline.load(run, index, q, "none", [1, 2])
    f = pa_dataset.field
    direct = read_table(full_run.run_dir / "tertiary" / "data" / "boom_final",
                        filter=(f("variable") == "ws") & (f("stat") == "mean") & (f("variant") == "none")
                        & (f("boom") == 2)).sort_values("slot")
    np.testing.assert_array_equal(data.curves[2].y, direct["value"].to_numpy())
    assert timeline.night(run, index) is not None
    stability = timeline.stability(run, index)
    assert stability is not None and stability[1] == [n for n, _ in run.cfg.post.stability.classes]
