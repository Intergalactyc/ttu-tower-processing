import threading

import numpy as np
import pandas as pd
import pytest

from ttu_tower.constants import HEIGHTS
from ttu_tower.flags import FlagStore
from ttu_tower.timegrid import SAMPLES_PER_SLOT
from ttu_tower.viewer import profiles, qcsummary, timeline
from ttu_tower.viewer.catalog import Quantity
from ttu_tower.viewer.fragments import FragmentIndex, alias_bounds_variables
from ttu_tower.viewer.run import RunHandle
from ttu_tower.viewer.timeline import Curve, TimelineData


def test_per_slot_samples_match_a_brute_force_count():
    rng = np.random.default_rng(3)
    slot_a, n_slots, rows_n = 100, 6, 3
    lo, hi = (slot_a - 1) * SAMPLES_PER_SLOT, (slot_a + n_slots + 1) * SAMPLES_PER_SLOT
    for _ in range(100):
        starts, ends, rows = [], [], []
        for r in range(rows_n):
            points = np.sort(rng.choice(np.arange(lo, hi, 991), 2 * rng.integers(1, 4), replace=False))
            starts += list(points[0::2])
            ends += list(points[1::2])
            rows += [r] * (points.size // 2)
        starts, ends, rows = (np.array(v, dtype=np.int64) for v in (starts, ends, rows))
        expected = np.zeros((rows_n, n_slots), dtype=np.int64)
        for s, e, r in zip(starts, ends, rows):
            for k in range(n_slots):
                s0 = (slot_a + k) * SAMPLES_PER_SLOT
                expected[r, k] += max(0, min(e, s0 + SAMPLES_PER_SLOT) - max(s, s0))
        assert (qcsummary.per_slot_samples(starts, ends, rows, rows_n, slot_a, n_slots) == expected).all()


def test_flag_counts_match_the_flag_store(primary_run):
    run = RunHandle.from_dir(primary_run.run_dir)
    index = FragmentIndex(run.run_dir)
    counts = qcsummary.flag_counts(run, index, run.booms)
    boom = run.booms[0]
    slots = np.arange(*run.period, dtype=np.int64)
    starts = slots * SAMPLES_PER_SLOT
    store = FlagStore(alias_bounds_variables(index.read("flags", booms=[boom])))
    assert counts[boom]
    for (test, variable), per_slot in counts[boom].items():
        stored = store._count_key((test, boom, variable), starts, starts + SAMPLES_PER_SLOT)
        assert (per_slot == stored).all(), (test, variable)
    assert ("bounds", "ue") in counts[boom]  # the raw-named bounds rows, under their earth names


def test_availability_shares_add_up_and_flags_are_percent(primary_run):
    run = RunHandle.from_dir(primary_run.run_dir)
    index = FragmentIndex(run.run_dir)
    slots, image = timeline.availability(run, index, run.booms)
    data = qcsummary.stored(run, index, run.booms)
    table = qcsummary.availability_table(slots, image, run.booms, data, None)
    shares = table[["computed %", "no data %", "no file %", "missing %"]].sum(axis=1)
    assert np.allclose(shares, 100.0) and (table["slots"] == slots.size).all()
    flagged = qcsummary.flag_table(slots, image, run.booms, qcsummary.flag_counts(run, index, run.booms), "w", None)
    assert {"bounds", "spike", "direction", "bounce"} <= set(flagged.columns)  # boom-level tests count toward w
    assert (flagged.fillna(0) >= 0).all().all() and (flagged.fillna(0) <= 100).all().all()
    assert "direction" not in qcsummary.flag_table(slots, image, run.booms, {}, "p", None).columns


def _quantity(variable="ws", circular=False) -> Quantity:
    return Quantity(key=f"boom_final|{variable}|mean", table="boom_final", variable=variable, stat="mean",
                    kind="boom", variants=("none",), label=variable, unit="m/s", group="Wind", circular=circular)


def _profile_data(alpha=0.2, n=40):
    """ws = 5 (z/10)^α at every boom, with a known spread about it."""
    slots = np.arange(1000, 1000 + n, dtype=np.int64)
    noise = np.where(slots % 2 == 0, 1.0, -1.0)
    curves = {b: Curve(slots=slots, y=5.0 * (HEIGHTS[b] / 10.0) ** alpha + noise) for b in HEIGHTS}
    return TimelineData(quantity=_quantity(), variant="none", curves=curves), slots


def test_profile_stats_give_mean_sigma_or_median_mad():
    data, _ = _profile_data()
    stats = profiles.profile_stats(data, list(HEIGHTS), None, None, "mean")
    assert list(stats["height"]) == sorted(HEIGHTS.values())
    expected = [5.0 * (z / 10.0) ** 0.2 for z in stats["height"]]
    assert np.allclose(stats["centre"], expected) and np.allclose(stats["spread"], 1.0)
    median = profiles.profile_stats(data, list(HEIGHTS), None, None, "median")
    assert np.allclose(median["centre"], expected) and np.allclose(median["spread"], 1.0)


def test_power_law_fit_recovers_the_shear_exponent():
    data, _ = _profile_data(alpha=0.23)
    stats = profiles.profile_stats(data, list(HEIGHTS), None, None, "mean")
    result, shown = profiles.fit_profile("power law", stats["height"].to_numpy(), stats["centre"].to_numpy())
    assert shown["α"] == pytest.approx(0.23, abs=1e-9) and result.n == len(HEIGHTS)
    assert profiles.fit_profile("power law", np.array([1.0, 2.0]), np.array([1.0, 2.0]))[0] is None
    assert profiles.default_fit(_quantity("ws")) == "power law" and profiles.default_fit(_quantity("t")) is None


def test_profiles_by_stability_split_the_slots():
    data, slots = _profile_data()
    names = ["unstable", "neutral", "stable"]
    stability = (Curve(slots=slots, y=(slots % 3).astype(float)), names)
    stats = profiles.profile_stats(data, list(HEIGHTS), None, stability, "mean", by_stability=True)
    assert list(dict.fromkeys(stats["group"])) == names  # in class order
    per_boom = stats.groupby("boom")["n"].sum()
    assert (per_boom == slots.size).all()


def test_pair_quantities_start_on_their_customary_pairs():
    from ttu_tower.viewer.ui.panel import default_pairs
    everything = tuple((a, b) for a in range(1, 11) for b in range(a + 1, 11))
    rib = Quantity(key="pairs|rib", table="pairs", variable="rib", stat=None, kind="pair", variants=("none",),
                   label="Ri_b", unit="", group="Stability", pairs=everything)
    assert default_pairs(rib) == [(2, 4)]
    veer_pairs = tuple((b, 4) for b in range(1, 11))
    veer = Quantity(key="pairs|veer", table="pairs", variable="veer", stat=None, kind="pair", variants=("none",),
                    label="veer", unit="deg", group="Stability", pairs=veer_pairs)
    assert default_pairs(veer) == [p for p in veer_pairs if p != (4, 4)]


def test_a_superseded_job_still_queued_never_runs(qtbot):
    pytest.importorskip("pytestqt")
    from ttu_tower.viewer.ui.worker import JobRunner
    runner = JobRunner(max_threads=1)
    gate, ran, delivered = threading.Event(), [], []
    runner.submit(lambda: gate.wait(10), key="blocker")
    runner.submit(lambda: ran.append("old"), key="same", on_done=lambda _: delivered.append("old"))
    runner.submit(lambda: ran.append("new"), key="same", on_done=lambda _: delivered.append("new"))
    runner.submit(lambda: ran.append("urgent"), key="other", priority=10)
    gate.set()
    qtbot.waitUntil(lambda: not runner._callbacks, timeout=10_000)
    assert ran == ["urgent", "new"] and delivered == ["new"]
    runner.wait()


def test_tau_choices_split_by_the_cospectrum_they_came_from(primary_run):
    stored = qcsummary.Stored(
        slots=np.arange(10, dtype=np.int64), coverage={},
        filter_log=pd.DataFrame(columns=["slot", "boom", "variant", "group", "criterion"]),
        tau=pd.DataFrame({"slot": np.arange(10), "boom": 1, "variant": "mrd",
                          "source": ["momentum"] * 4 + ["heat"] * 3 + ["fallback", "heat", "none"],
                          "source_status": ["found"] * 7 + ["fallback", "capped", "none"]}))
    image = np.full((1, 10), qcsummary.COMPUTED, dtype=np.int8)
    table = qcsummary.tau_table(stored.slots, image, [1], stored, "mrd", None)
    assert table.iloc[0].to_dict() == {"found · momentum": 40.0, "found · heat": 30.0, "capped · heat": 10.0,
                                       "fallback": 10.0, "none": 10.0}


def test_whole_numbers_print_in_full():
    from ttu_tower.viewer.ui.tables import text
    assert text(10123.0) == "10123" and text(0.05123) == "0.05123" and text(1.5e7 + 0.5) == "1.5e+07"
