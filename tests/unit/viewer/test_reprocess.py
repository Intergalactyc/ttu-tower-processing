import threading

import numpy as np
import pyarrow.dataset as pa_dataset
import pytest

from ttu_tower.timegrid import SAMPLES_PER_HALF_HOUR, SAMPLES_PER_SLOT
from ttu_tower.viewer import reprocess as reprocess_module
from ttu_tower.viewer.fragments import FragmentIndex
from ttu_tower.viewer.reprocess import Reprocessor, slot_window
from ttu_tower.viewer.run import RunHandle

from viewer_fixtures import BOUNDS_INDEX, FAULT_HALF_HOUR, PRIMARY_MISSING, TS_SPIKE_INDEX, W_SPIKE_INDEX

_FAMILY = {"ue": "momentum", "vn": "momentum", "w": "momentum", "ts": "ts", "vpts": "ts"}


@pytest.fixture(scope="module")
def run(primary_run):
    return RunHandle.from_dir(primary_run.run_dir)


def test_qc_series_reproduce_every_stored_slot_mean(run):
    """The key correctness check: the QC'd window is exactly what primary averaged."""
    index = FragmentIndex(run.run_dir)
    means = index.read("means", booms=[1], filter=pa_dataset.field("stat") == "mean")
    rp = Reprocessor(run)
    checked = 0
    for k, group in means.groupby("slot"):
        g0 = int(k) * SAMPLES_PER_SLOT
        win = rp.qc_applied(1, g0, SAMPLES_PER_SLOT)
        for _, row in group.iterrows():
            var = str(row["variable"])
            if var in _FAMILY:
                x, m = win.series[var], win.family_masks[_FAMILY[var]]
            elif var in ("t", "rh", "p"):
                x, m = win.slow_smoothed[var]
            else:
                continue  # ws/wd means are derived from ue/vn, not a series of their own
            value = x[m].mean() if m.any() else np.nan
            if np.isnan(row["value"]):
                assert np.isnan(value) or not m.any() or m.mean() < run.cfg.qc.min_coverage
            else:
                assert value == pytest.approx(row["value"], rel=1e-12, abs=1e-12), (k, var)
                checked += 1
    assert checked > 50


def _fault_window(run):
    rp = Reprocessor(run)
    g0 = FAULT_HALF_HOUR * SAMPLES_PER_HALF_HOUR
    return rp, g0


def test_as_measured_keeps_what_qc_removes(run):
    rp, g0 = _fault_window(run)
    am = rp.as_measured(1, g0, SAMPLES_PER_HALF_HOUR)
    qc = rp.qc_applied(1, g0, SAMPLES_PER_HALF_HOUR)

    assert np.isfinite(am.series["ue"][BOUNDS_INDEX])
    assert qc.removed["bounds"]["ue"][BOUNDS_INDEX]
    assert qc.removed["bounds"]["vn"][BOUNDS_INDEX]

    assert qc.removed["spike"]["w"][W_SPIKE_INDEX]
    assert qc.filled["w"][W_SPIKE_INDEX]  # a lone removed sample comes back via the <= 1-s fill
    # ue/vn lose the sample by coupling, or by their own despike where tilt leaks some of w into them
    for var in ("ue", "vn"):
        by_own = var in qc.removed["spike"] and qc.removed["spike"][var][W_SPIKE_INDEX]
        by_coupling = var in qc.removed["coupled"] and qc.removed["coupled"][var][W_SPIKE_INDEX]
        assert by_own or by_coupling, var

    assert qc.removed["spike"]["ts"][TS_SPIKE_INDEX]
    assert qc.removed["spike"]["vpts"][TS_SPIKE_INDEX]
    assert abs(am.series["w"][W_SPIKE_INDEX] - qc.series["w"][W_SPIKE_INDEX]) > 3.0


def test_despike_band_brackets_the_kept_samples(run):
    rp, g0 = _fault_window(run)
    qc = rp.qc_applied(1, g0, SAMPLES_PER_HALF_HOUR)
    am = rp.as_measured(1, g0, SAMPLES_PER_HALF_HOUR)
    lo, hi = qc.despike_band["ts"]
    assert not (lo[TS_SPIKE_INDEX] <= am.series["ts"][TS_SPIKE_INDEX] <= hi[TS_SPIKE_INDEX])
    inside = (am.series["ts"] >= lo) & (am.series["ts"] <= hi)
    assert np.nanmean(inside) > 0.99


def test_sonic_frame_differs_from_earth_frame(run):
    rp, g0 = _fault_window(run)
    sonic = rp.as_measured(1, g0, SAMPLES_PER_HALF_HOUR, frame="sonic")
    earth = rp.as_measured(1, g0, SAMPLES_PER_HALF_HOUR)
    assert set(sonic.series) >= {"u", "v", "w"} and "ue" not in sonic.series
    speed_sonic = np.hypot(sonic.series["u"], sonic.series["v"])
    speed_earth = np.hypot(earth.series["ue"], earth.series["vn"])
    assert np.nanmedian(np.abs(speed_sonic - speed_earth)) < 0.5  # rotation preserves speed, tilt barely changes it


def test_missing_half_hour_is_reported_and_nan(run):
    rp = Reprocessor(run)
    h = PRIMARY_MISSING[0]
    g0, n = h * SAMPLES_PER_HALF_HOUR - 1000, SAMPLES_PER_HALF_HOUR + 2000
    qc = rp.qc_applied(1, g0, n)
    assert h in qc.missing_half_hours
    assert np.isnan(qc.series["ue"][1000:1000 + SAMPLES_PER_HALF_HOUR]).all()
    am = rp.as_measured(1, g0, n)
    assert h in am.missing_half_hours


def test_unexcised_mode_uses_the_unexcised_series(run):
    rp, g0 = _fault_window(run)
    un = rp.qc_applied(1, g0, SAMPLES_PER_HALF_HOUR, variant="mrd_unexcised")
    assert un.mode == "unexcised"
    assert np.isfinite(un.series["w"][W_SPIKE_INDEX])


def test_warm_calls_do_not_reload_files(run, monkeypatch):
    calls = []
    real = reprocess_module.load_boom
    monkeypatch.setattr(reprocess_module, "load_boom", lambda *a, **k: calls.append(a) or real(*a, **k))
    rp = Reprocessor(run)
    g0, n = slot_window(3 * FAULT_HALF_HOUR + 1)
    rp.qc_applied(1, g0, n)
    cold = len(calls)
    rp.qc_applied(1, g0, n)
    assert cold > 0 and len(calls) == cold


def test_concurrent_requests_compute_stage_b_once(run, monkeypatch):
    computed = []
    real = reprocess_module.stage_b

    def counting(*args, **kwargs):
        computed.append(args[1])
        return real(*args, **kwargs)

    monkeypatch.setattr(reprocess_module, "stage_b", counting)
    rp = Reprocessor(run)
    threads = [threading.Thread(target=rp.stage_b, args=(1, FAULT_HALF_HOUR)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert computed == [FAULT_HALF_HOUR]


def test_streamwise_frame_has_zero_mean_v_in_every_slot(run):
    from ttu_tower.viewer.reprocess import to_streamwise

    rp = Reprocessor(run)
    g0 = FAULT_HALF_HOUR * SAMPLES_PER_HALF_HOUR
    win = rp.qc_applied(1, g0, SAMPLES_PER_HALF_HOUR)
    rotated, bearings = to_streamwise(win)
    assert "ue" not in rotated.series and {"u", "v"} <= set(rotated.series)
    assert set(bearings) == {3 * FAULT_HALF_HOUR + i for i in range(3)}
    for i in range(3):
        sl = slice(i * SAMPLES_PER_SLOT, (i + 1) * SAMPLES_PER_SLOT)
        ok = rotated.masks["u"][sl]
        if not ok.any():  # nothing usable: rotated by the mean of its finite samples instead
            ok = np.isfinite(rotated.series["u"][sl])
        assert abs(rotated.series["v"][sl][ok].mean()) < 1e-9
        assert rotated.series["u"][sl][ok].mean() > 0
    # rotation preserves the horizontal speed
    speed_earth = np.hypot(win.series["ue"], win.series["vn"])
    speed_stream = np.hypot(rotated.series["u"], rotated.series["v"])
    np.testing.assert_allclose(speed_stream, speed_earth, rtol=1e-12)
    assert "u" in rotated.removed["spike"] and "ue" not in rotated.removed["spike"]
