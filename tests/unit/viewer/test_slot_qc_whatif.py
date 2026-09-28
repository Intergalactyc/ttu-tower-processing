import numpy as np
import pandas as pd
import pyarrow.dataset as pa_dataset
import pytest

from ttu_tower.io.store import read_table
from ttu_tower.timegrid import SAMPLES_PER_SLOT
from ttu_tower.viewer import profiledata, slotdata, slotqc, timeline, whatif
from ttu_tower.viewer.catalog import build_catalog
from ttu_tower.viewer.fragments import FragmentIndex
from ttu_tower.viewer.run import RunHandle

from viewer_fixtures import FAULT_HALF_HOUR, FULL_BOOMS, SPIKY_BOOM, SPIKY_SLOT


def _computed_slots(run_dir, boom):
    sb = read_table(run_dir / "primary" / "data" / "slot_boom")
    return sorted(sb[(sb["boom"] == boom) & (sb["status"].astype(str) == "computed")]["slot"].tolist())


# --- primary-only run (fast) ------------------------------------------------------------

@pytest.fixture(scope="module")
def primary(primary_run):
    run = RunHandle.from_dir(primary_run.run_dir)
    return run, FragmentIndex(run.run_dir)


def test_coverage_ladder_is_the_coverage_table(primary):
    _, index = primary
    k = 3 * FAULT_HALF_HOUR + 1
    bundle = slotdata.load_slot(index, k, 1, booms=[1])
    ladder = slotqc.coverage_ladder(bundle.table("coverage"))
    for r in bundle.table("coverage").itertuples(index=False):
        assert ladder.loc[r.variable, r.layer] == r.fraction
    assert np.isnan(ladder.loc["momentum", "present"])


def test_flag_fractions_see_the_injected_faults(primary):
    _, index = primary
    k = 3 * FAULT_HALF_HOUR + 1
    g_lo, g_hi = slotqc.windows(k)["detection window (80 min)"]
    flags = index.flags_window(1, g_lo, g_hi)
    fractions = slotqc.flag_fractions(flags, 1, g_lo, g_hi)
    assert fractions.loc["bounds", "ue"] > 0  # the out-of-bounds u, under its earth-frame name
    assert fractions.loc["spike", "ts"] > 0 and fractions.loc["spike", "w"] > 0
    assert np.isnan(fractions.loc["resolution", "t"])  # a test that never applies there


def test_slot_qc_checks_agree_with_the_stored_slot_flags(primary):
    run, index = primary
    checked = 0
    for k in _computed_slots(run.run_dir, 1):
        bundle = slotdata.load_slot(index, k, 1, booms=[1])
        checks = slotqc.slot_qc_checks(bundle.table("slot_qc"), run.cfg.qc)
        s0 = SAMPLES_PER_SLOT * k
        fractions = slotqc.flag_fractions(index.flags_window(1, s0, s0 + SAMPLES_PER_SLOT), 1, s0, s0 + SAMPLES_PER_SLOT)
        for r in checks[checks["test"].isin(["skew", "kurt"])].itertuples(index=False):
            assert r.tripped == (fractions.loc[r.test, r.variable] == 1.0), (k, r)
            checked += 1
    assert checked > 50


def test_whatif_rejects_inconsistent_parameters(primary_run):
    cfg = primary_run.cfg.secondary
    assert whatif.problems(cfg) == []
    bad = whatif.edited(cfg, detection={"min_tau_s": 600.0}, selection={"fallback_tau_s": 300.0})
    assert bad.detection.min_tau_s == 600.0 and cfg.detection.min_tau_s != 600.0
    assert whatif.problems(bad)


# --- full run (slow) --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def full(full_run):
    run = RunHandle.from_dir(full_run.run_dir)
    return run, FragmentIndex(run.run_dir)


@pytest.mark.slow
def test_whatif_with_the_runs_parameters_reproduces_every_stored_tau(full):
    run, index = full
    stored = read_table(run.run_dir / "secondary" / "data" / "tau_selected")
    stored = stored[stored["variant"].astype(str) != "naive"]
    checked = 0
    for k in sorted(set(stored["slot"])):
        across = slotdata.load_across(index, k, FULL_BOOMS)
        compared = whatif.compare(across, run.cfg.secondary)
        assert not compared["changed"].any(), compared[compared["changed"]]
        _, new_sel = whatif.rerun(across, run.cfg.secondary)
        old = stored[stored["slot"] == k].astype({"variant": object, "source": object, "source_status": object})
        merged = old.merge(new_sel, on=["slot", "boom", "variant"], suffixes=("", "_new"))
        assert len(merged) == len(old) > 0
        assert (merged["source"] == merged["source_new"]).all()
        assert (merged["source_status"] == merged["source_status_new"]).all()
        assert np.array_equal(merged["tau_s"].to_numpy(), merged["tau_s_new"].to_numpy(), equal_nan=True)
        checked += len(merged)
    assert checked > 50


@pytest.mark.slow
def test_whatif_with_a_stricter_test_changes_tau_and_its_statistics(full):
    run, index = full
    strict = whatif.edited(run.cfg.secondary, detection={"peak_significance_se": 50.0})
    changed = 0
    for k in _computed_slots(run.run_dir, 1)[:6]:
        compared = whatif.compare(slotdata.load_across(index, k, FULL_BOOMS), strict)
        changed += int(compared["changed"].sum())
        moved = compared[compared["selected_tau_new"] != compared["selected_tau_stored"]]
        assert (moved["ustar_new"] != moved["ustar_stored"]).all()
    assert changed > 0


@pytest.mark.slow
def test_whatif_ustar_at_the_stored_tau_is_the_stored_ustar(full):
    run, index = full
    across = slotdata.load_across(index, SPIKY_SLOT - 3, FULL_BOOMS)
    compared = whatif.compare(across, run.cfg.secondary)
    stats = across.table("boom_stats")
    for r in compared.itertuples(index=False):
        row = stats[(stats["boom"] == r.boom) & (stats["variant"] == r.variant) & (stats["variable"] == "ustar")]
        assert r.ustar_stored == pytest.approx(row["value"].iloc[0], rel=1e-12, nan_ok=True)


@pytest.mark.slow
def test_filter_gates_reproduce_the_stored_filter_log(full):
    run, index = full
    checked = failing = 0
    for boom in FULL_BOOMS:
        for k in _computed_slots(run.run_dir, boom):
            bundle = slotdata.load_slot(index, k, boom, booms=FULL_BOOMS)
            g_lo, g_hi = slotqc.windows(k)["detection window (80 min)"]
            gates = slotqc.filter_gates(k, boom, bundle.table("means"), bundle.table("slow"),
                                        bundle.table("boom_stats"), bundle.table("coverage"),
                                        bundle.table("tau_selected"), index.flags_window(boom, g_lo, g_hi),
                                        run.cfg.tertiary)
            assert slotqc.logged_failures(gates) == slotqc.stored_failures(bundle.table("filter_log"), boom), (k, boom)
            checked += 1
            failing += bool(gates["failed"].any())
    assert checked >= 20 and failing > 0


@pytest.mark.slow
def test_filtered_overlay_is_the_secondary_value_wherever_tertiary_nand_it(full):
    run, index = full
    catalog = build_catalog(run)
    f = pa_dataset.field
    for key, variant, table in (("boom_final|sigma_w|", "mrd", "boom_stats"), ("boom_final|ws|mean", "none", "means")):
        q = catalog[key]
        overlay = timeline.filtered(run, index, q, variant, FULL_BOOMS)
        final = timeline.load(run, index, q, variant, FULL_BOOMS)
        stage = "secondary" if table == "boom_stats" else "primary"
        flt = (f("variable") == q.variable) & (f("stat").is_null() if q.stat is None else f("stat") == q.stat)
        if table == "boom_stats":
            flt = flt & (f("variant") == variant)
        pre = read_table(run.run_dir / stage / "data" / table, filter=flt)
        for boom in FULL_BOOMS:
            fin = pd.Series(final.curves[boom].y, index=final.curves[boom].slots)
            rows = pre[pre["boom"] == boom].set_index("slot")["value"]
            expected = rows[rows.notna() & fin.reindex(rows.index).isna()].sort_index()
            got = overlay.curves[boom]
            np.testing.assert_array_equal(got.slots, expected.index.to_numpy())
            np.testing.assert_array_equal(got.y, expected.to_numpy())
    sigma = timeline.filtered(run, index, catalog["boom_final|sigma_w|"], "mrd", FULL_BOOMS)
    assert SPIKY_SLOT in sigma.curves[SPIKY_BOOM].slots
    assert "spike w" in sigma.reasons[(SPIKY_SLOT, SPIKY_BOOM)]


@pytest.mark.slow
def test_tau_status_marks_what_tau_final_says(full):
    run, index = full
    marked = timeline.tau_status(run, index, "mrd", FULL_BOOMS)
    tau_final = read_table(run.run_dir / "tertiary" / "data" / "tau_final")
    tau_final = tau_final[tau_final["variant"].astype(str) == "mrd"]
    for boom in FULL_BOOMS:
        rows = tau_final[(tau_final["boom"] == boom)
                         & tau_final["source_status"].astype(str).isin(timeline.TAU_MARKED)].sort_values("slot")
        slots, status = marked[boom]
        np.testing.assert_array_equal(slots, rows["slot"].to_numpy())
        assert list(status) == list(rows["source_status"].astype(str))


@pytest.mark.slow
def test_profile_ghosts_the_filtered_boom_and_redraws_the_stored_fits(full):
    run, index = full
    across = slotdata.load_across(index, SPIKY_SLOT, FULL_BOOMS)
    sigma_w = next(p for p in profiledata.PROFILE_QUANTITIES if p.key == "sigma_w")
    points = profiledata.profile_points(across, sigma_w, "mrd").set_index("boom")
    assert points.loc[SPIKY_BOOM, "filtered"] and np.isfinite(points.loc[SPIKY_BOOM, "prefilter"])
    assert "spike w" in points.loc[SPIKY_BOOM, "reasons"]
    other = next(b for b in FULL_BOOMS if b != SPIKY_BOOM)
    assert not points.loc[other, "filtered"] and np.isfinite(points.loc[other, "value"])

    k = _computed_slots(run.run_dir, 1)[4]
    across = slotdata.load_across(index, k, FULL_BOOMS)
    ws = next(p for p in profiledata.PROFILE_QUANTITIES if p.key == "ws")
    points = profiledata.profile_points(across, ws, "mrd")
    fits = {f.name: f for f in profiledata.profile_fits(across, "ws", points, "mrd", run.cfg.tertiary.fits)}
    assert "alpha" in fits and "refit" not in fits["alpha"].text
    alpha = across.table("profile")
    alpha = alpha[(alpha["variable"] == "alpha") & (alpha["variant"] == "none")]["value"].iloc[0]
    f = fits["alpha"]
    np.testing.assert_allclose(np.diff(np.log(f.y)) / np.diff(np.log(f.z)), alpha)
    pairs = profiledata.pair_table(across, run.cfg.post.stability.pair)
    assert len(pairs) == 1 and set(pairs.columns) == {"boom", "boom2", "rib", "lapse_vpt"}
